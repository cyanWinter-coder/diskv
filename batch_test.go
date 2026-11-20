package diskv

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"testing"
)

// failingReader fails after delivering prefix bytes.
type failingReader struct {
	prefix []byte
	n      int
	err    error
}

func (r *failingReader) Read(p []byte) (int, error) {
	if r.n >= len(r.prefix) {
		return 0, r.err
	}
	n := copy(p, r.prefix[r.n:])
	r.n += n
	if r.n >= len(r.prefix) {
		return n, r.err
	}
	return n, nil
}

func newTestDiskv(t *testing.T, opts Options) *Diskv {
	if opts.BasePath == "" {
		opts.BasePath = "test-batch-data"
	}
	d := New(opts)
	t.Cleanup(func() { d.EraseAll() })
	return d
}

func allKeys(d *Diskv) []string {
	keys := []string{}
	for k := range d.Keys(nil) {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

func TestWriteBatchBasic(t *testing.T) {
	d := newTestDiskv(t, Options{CacheSizeMax: 4096})
	items := []BatchItem{
		{Key: "k1", Val: []byte("value-one")},
		{Key: "k2", Val: []byte("value-two")},
		{Key: "k3", Val: []byte("value-three")},
	}
	if err := d.WriteBatch("batch-1", items); err != nil {
		t.Fatalf("WriteBatch: %s", err)
	}
	for _, it := range items {
		got, err := d.Read(it.Key)
		if err != nil {
			t.Fatalf("Read(%q): %s", it.Key, err)
		}
		if !bytes.Equal(got, it.Val) {
			t.Fatalf("Read(%q): got %q want %q", it.Key, got, it.Val)
		}
	}
	if got := allKeys(d); len(got) != 3 || got[0] != "k1" || got[2] != "k3" {
		t.Fatalf("Keys: %v", got)
	}
	// Staging must be cleaned up after commit (only the retained .committed
	// registry may remain).
	root := d.stagingRoot()
	if entries, err := ioutil.ReadDir(root); err != nil || len(entries) != 1 || entries[0].Name() != batchRegistryDirName {
		t.Fatalf("unexpected staging contents: %v (err=%v)", entries, err)
	}
}

func TestWriteBatchIndexCoupled(t *testing.T) {
	idx := &BTreeIndex{}
	d := newTestDiskv(t, Options{
		Index:     idx,
		IndexLess: func(a, b string) bool { return a < b },
	})
	items := []BatchItem{
		{Key: "a", Val: []byte("1")},
		{Key: "b", Val: []byte("2")},
		{Key: "c", Val: []byte("3")},
	}
	if err := d.WriteBatch("b1", items); err != nil {
		t.Fatal(err)
	}
	if got := idx.Keys("", 10); len(got) != 3 {
		t.Fatalf("index keys: %v", got)
	}

	// A failing batch must not change the index at all.
	bad := []BatchStreamItem{
		{Key: "d", Reader: strings.NewReader("ok")},
		{Key: "e", Reader: &failingReader{prefix: []byte("part"), err: errors.New("disk exploded")}},
	}
	err := d.WriteBatchStream("b2", bad)
	if err == nil {
		t.Fatal("expected batch error")
	}
	var be *BatchError
	if !errors.As(err, &be) {
		t.Fatalf("expected *BatchError, got %T: %v", err, err)
	}
	if len(be.Errors) != 1 || be.Errors[0].Key != "e" {
		t.Fatalf("unexpected batch error detail: %+v", be)
	}
	if got := idx.Keys("", 10); len(got) != 3 {
		t.Fatalf("index changed after failed batch: %v", got)
	}
	if d.Has("d") || d.Has("e") {
		t.Fatal("failed batch keys visible")
	}
	if _, err := os.Stat(d.stagingDirWithLock("b2")); !os.IsNotExist(err) {
		t.Fatalf("temp files not cleaned: %v", err)
	}
}

func TestWriteBatchFailureRollbackNewKeys(t *testing.T) {
	d := newTestDiskv(t, Options{})
	bad := []BatchStreamItem{
		{Key: "good", Reader: strings.NewReader("payload")},
		{Key: "bad", Reader: &failingReader{prefix: []byte("x"), err: errors.New("boom")}},
	}
	if err := d.WriteBatchStream("b", bad); err == nil {
		t.Fatal("expected failure")
	}
	if d.Has("good") || d.Has("bad") {
		t.Fatal("rolled-back keys still present")
	}
	if len(allKeys(d)) != 0 {
		t.Fatal("store not empty after rollback")
	}
}

func TestWriteBatchFailureRollbackOverwrite(t *testing.T) {
	d := newTestDiskv(t, Options{CacheSizeMax: 4096})
	if err := d.Write("keep", []byte("original")); err != nil {
		t.Fatal(err)
	}
	if err := d.Write("other", []byte("other-original")); err != nil {
		t.Fatal(err)
	}
	// Warm the cache with the original value.
	if _, err := d.Read("keep"); err != nil {
		t.Fatal(err)
	}

	bad := []BatchStreamItem{
		{Key: "keep", Reader: strings.NewReader("new-value")},
		{Key: "boom", Reader: &failingReader{err: errors.New("nope")}},
	}
	if err := d.WriteBatchStream("overwrite", bad); err == nil {
		t.Fatal("expected failure")
	}

	got, err := d.Read("keep")
	if err != nil {
		t.Fatalf("Read after rollback: %s", err)
	}
	if string(got) != "original" {
		t.Fatalf("overwrite not rolled back: %q", got)
	}
	if !d.isCached("keep") || string(d.cache["keep"]) != "original" {
		t.Fatal("cache lost the original value on rollback")
	}
	if got, err := d.Read("other"); err != nil || string(got) != "other-original" {
		t.Fatalf("unrelated key disturbed: %q %v", got, err)
	}
}

func TestWriteBatchIdempotent(t *testing.T) {
	d := newTestDiskv(t, Options{})
	first := []BatchItem{{Key: "k", Val: []byte("first")}}
	if err := d.WriteBatch("same-id", first); err != nil {
		t.Fatal(err)
	}
	// Same batch ID with a different payload must be a no-op.
	second := []BatchItem{{Key: "k", Val: []byte("second")}}
	if err := d.WriteBatch("same-id", second); err != nil {
		t.Fatalf("re-submit: %s", err)
	}
	got, err := d.Read("k")
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != "first" {
		t.Fatalf("duplicate batch ID re-applied payload: %q", got)
	}
}

func TestWriteBatchConcurrentSameKey(t *testing.T) {
	d := newTestDiskv(t, Options{CacheSizeMax: 1 << 20})
	const goroutines = 16
	var wg sync.WaitGroup
	wg.Add(goroutines)
	errs := make(chan error, goroutines)
	for g := 0; g < goroutines; g++ {
		g := g
		go func() {
			defer wg.Done()
			val := []byte(fmt.Sprintf("gen-%02d", g))
			errs <- d.WriteBatch(fmt.Sprintf("batch-%02d", g), []BatchItem{{Key: "race", Val: val}})
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			t.Fatalf("batch: %s", err)
		}
	}

	// Concurrent readers must only ever see one complete value, never a torn
	// or length-mismatched buffer.
	stop := make(chan struct{})
	var readerWg sync.WaitGroup
	for r := 0; r < 8; r++ {
		readerWg.Add(1)
		go func() {
			defer readerWg.Done()
			for {
				select {
				case <-stop:
					return
				default:
					got, err := d.Read("race")
					if err != nil {
						t.Errorf("read: %s", err)
						return
					}
					if len(got) != len("gen-00") || !strings.HasPrefix(string(got), "gen-") {
						t.Errorf("torn value: %q", got)
						return
					}
				}
			}
		}()
	}

	// Keep overwriting while readers run.
	for g := goroutines; g < goroutines*3; g++ {
		val := []byte(fmt.Sprintf("gen-%02d", g))
		if err := d.WriteBatch(fmt.Sprintf("batch-%02d", g), []BatchItem{{Key: "race", Val: val}}); err != nil {
			t.Fatal(err)
		}
	}
	close(stop)
	readerWg.Wait()
}

func TestWriteBatchValidation(t *testing.T) {
	d := newTestDiskv(t, Options{})
	if err := d.WriteBatch("x", nil); err != errEmptyBatch {
		t.Fatalf("empty batch: got %v", err)
	}
	if err := d.WriteBatch("", []BatchItem{{Key: "k", Val: []byte("v")}}); err != errEmptyBatchID {
		t.Fatalf("empty batch ID: got %v", err)
	}
	for _, bad := range []string{"a/b", "..", "a\\b"} {
		if err := d.WriteBatch(bad, []BatchItem{{Key: "k", Val: []byte("v")}}); err != errBadBatchID {
			t.Fatalf("bad batch ID %q: got %v", bad, err)
		}
	}
	err := d.WriteBatch("dup", []BatchItem{
		{Key: "k", Val: []byte("a")},
		{Key: "k", Val: []byte("b")},
	})
	if err == nil || !strings.Contains(err.Error(), "duplicate") {
		t.Fatalf("duplicate key: got %v", err)
	}
	err = d.WriteBatch("emptykey", []BatchItem{{Key: "", Val: []byte("v")}})
	var be *BatchError
	if !errors.As(err, &be) || be.Errors[0].Err != errEmptyKey {
		t.Fatalf("empty key: got %v", err)
	}
	if len(allKeys(d)) != 0 {
		t.Fatal("validation failures must not write anything")
	}
}

func TestWriteBatchCompression(t *testing.T) {
	d := newTestDiskv(t, Options{
		CacheSizeMax: 4096,
		Compression:  NewGzipCompression(),
	})
	items := []BatchItem{
		{Key: "c1", Val: bytes.Repeat([]byte("compressible-"), 64)},
		{Key: "c2", Val: []byte("small")},
	}
	if err := d.WriteBatch("gz", items); err != nil {
		t.Fatal(err)
	}
	for _, it := range items {
		got, err := d.Read(it.Key)
		if err != nil {
			t.Fatalf("Read: %s", err)
		}
		if !bytes.Equal(got, it.Val) {
			t.Fatalf("compressed round-trip mismatch for %q", it.Key)
		}
	}
}

func TestWriteBatchStagingSkippedByKeys(t *testing.T) {
	d := newTestDiskv(t, Options{Transform: blockTransform(2)})
	if err := d.WriteBatch("visible", []BatchItem{
		{Key: "ab01", Val: []byte("v")},
	}); err != nil {
		t.Fatal(err)
	}
	// Plant orphan staging content as a crash would leave it.
	sdir := d.stagingDirWithLock("orphan")
	if err := os.MkdirAll(filepath.Join(sdir, batchTempDirName, "ab"), d.PathPerm); err != nil {
		t.Fatal(err)
	}
	if err := ioutil.WriteFile(
		filepath.Join(sdir, batchTempDirName, "ab", "ab99"+batchTempSuffix),
		[]byte("orphan"), d.FilePerm); err != nil {
		t.Fatal(err)
	}
	if err := ioutil.WriteFile(filepath.Join(sdir, batchCheckpointName), []byte("{}"), d.FilePerm); err != nil {
		t.Fatal(err)
	}
	keys := allKeys(d)
	if len(keys) != 1 || keys[0] != "ab01" {
		t.Fatalf("staging internals leaked into Keys(): %v", keys)
	}
}

func TestRecoveryRollbackUncommittedBatch(t *testing.T) {
	base := "test-batch-recovery-rollback"
	defer os.RemoveAll(base)
	opts := Options{BasePath: base}
	d := New(opts)
	if err := d.Write("old", []byte("old-value")); err != nil {
		t.Fatal(err)
	}
	batchID := "interrupted"
	sdir := d.stagingDirWithLock(batchID)

	// Simulate a crash mid-commit: old file backed up, new block staged and
	// partially installed, no commit marker.
	if err := os.MkdirAll(filepath.Join(sdir, batchBackupDirName), d.PathPerm); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(sdir, batchTempDirName), d.PathPerm); err != nil {
		t.Fatal(err)
	}
	oldFinal := d.completeFilename(d.transform("old"))
	if err := os.Rename(oldFinal, filepath.Join(sdir, batchBackupDirName, "old")); err != nil {
		t.Fatal(err)
	}
	if err := ioutil.WriteFile(filepath.Join(sdir, batchTempDirName, "old"+batchTempSuffix), []byte("new-block"), d.FilePerm); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(sdir, batchTempDirName, "fresh"), d.PathPerm); err != nil {
		t.Fatal(err)
	}
	if err := ioutil.WriteFile(filepath.Join(sdir, batchTempDirName, "fresh", "newkey"+batchTempSuffix), []byte("fresh-block"), d.FilePerm); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(d.finalPathForRel(filepath.Join("fresh", "newkey"))), d.PathPerm); err != nil {
		t.Fatal(err)
	}
	if err := ioutil.WriteFile(d.finalPathForRel(filepath.Join("fresh", "newkey")), []byte("fresh-block"), d.FilePerm); err != nil {
		t.Fatal(err)
	}

	d2 := New(opts) // reopen triggers recovery
	defer d2.EraseAll()

	got, err := d2.Read("old")
	if err != nil {
		t.Fatalf("old value not restored: %s", err)
	}
	if string(got) != "old-value" {
		t.Fatalf("restored wrong value: %q", got)
	}
	if d2.Has("newkey") {
		t.Fatal("new-key block survived rollback")
	}
	if _, err := os.Stat(sdir); !os.IsNotExist(err) {
		t.Fatal("staging directory not cleaned")
	}
}

func TestRecoveryForwardCommittedBatch(t *testing.T) {
	base := "test-batch-recovery-forward"
	defer os.RemoveAll(base)
	opts := Options{BasePath: base}
	d := New(opts)
	batchID := "committed-batch"
	sdir := d.stagingDirWithLock(batchID)
	if err := os.MkdirAll(filepath.Join(sdir, batchTempDirName), d.PathPerm); err != nil {
		t.Fatal(err)
	}
	// One block installed, one still in tmp, but the marker exists.
	if err := os.MkdirAll(d.pathFor(d.transform("a")), d.PathPerm); err != nil {
		t.Fatal(err)
	}
	if err := ioutil.WriteFile(d.completeFilename(d.transform("a")), []byte("val-a"), d.FilePerm); err != nil {
		t.Fatal(err)
	}
	if err := ioutil.WriteFile(filepath.Join(sdir, batchTempDirName, "b"+batchTempSuffix), []byte("val-b"), d.FilePerm); err != nil {
		t.Fatal(err)
	}
	cp := batchCheckpoint{BatchID: batchID, Items: []batchItemRecord{
		{Key: "a", Rel: "a"},
		{Key: "b", Rel: "b"},
	}}
	data, _ := json.Marshal(&cp)
	if err := ioutil.WriteFile(filepath.Join(sdir, batchCheckpointName), data, d.FilePerm); err != nil {
		t.Fatal(err)
	}
	if err := ioutil.WriteFile(filepath.Join(sdir, batchMarkerName), nil, d.FilePerm); err != nil {
		t.Fatal(err)
	}

	d2 := New(opts)
	defer d2.EraseAll()
	for k, want := range map[string]string{"a": "val-a", "b": "val-b"} {
		got, err := d2.Read(k)
		if err != nil {
			t.Fatalf("Read(%q): %s", k, err)
		}
		if string(got) != want {
			t.Fatalf("Read(%q): got %q want %q", k, got, want)
		}
	}
	if _, err := os.Stat(sdir); !os.IsNotExist(err) {
		t.Fatal("staging directory not cleaned after forward recovery")
	}

	// The recovered batch ID is known: a retry is a no-op.
	if err := d2.WriteBatch(batchID, []BatchItem{{Key: "a", Val: []byte("override")}}); err != nil {
		t.Fatal(err)
	}
	got, _ := d2.Read("a")
	if string(got) != "val-a" {
		t.Fatalf("recovered batch ID was re-applied: %q", got)
	}
}

func TestRecoveryTempDirLayout(t *testing.T) {
	base, tmp := "test-batch-td-base", "test-batch-td-tmp"
	defer os.RemoveAll(base)
	defer os.RemoveAll(tmp)
	opts := Options{BasePath: base, TempDir: tmp}
	d := New(opts)
	if err := d.WriteBatch("b", []BatchItem{{Key: "k", Val: []byte("v")}}); err != nil {
		t.Fatal(err)
	}
	// Per-batch staging is gone; only the retained idempotency registry stays.
	if entries, err := ioutil.ReadDir(d.stagingRoot()); err != nil || len(entries) != 1 || entries[0].Name() != batchRegistryDirName {
		t.Fatalf("unexpected staging contents: %v (err=%v)", entries, err)
	}
	if _, err := os.Stat(d.stagingDirWithLock("b")); !os.IsNotExist(err) {
		t.Fatal("batch staging directory not removed")
	}
	got, err := d.Read("k")
	if err != nil || string(got) != "v" {
		t.Fatalf("round trip with TempDir: %q %v", got, err)
	}
}

func TestWriteBatchString(t *testing.T) {
	d := newTestDiskv(t, Options{})
	if err := d.WriteBatchString("s", map[string]string{"a": "x", "b": "y"}); err != nil {
		t.Fatal(err)
	}
	for k, want := range map[string]string{"a": "x", "b": "y"} {
		got := d.ReadString(k)
		if got != want {
			t.Fatalf("%q: got %q want %q", k, got, want)
		}
	}
}

func TestRecoveryOrphanCleanup(t *testing.T) {
	base := "test-batch-orphans"
	defer os.RemoveAll(base)
	opts := Options{BasePath: base}
	d := New(opts)
	// Orphan staging from a crash before the commit decision: tmp blocks only,
	// no marker, and some blocks were installed into BasePath.
	sdir := d.stagingDirWithLock("orphan")
	if err := os.MkdirAll(filepath.Join(sdir, batchTempDirName, "ab"), d.PathPerm); err != nil {
		t.Fatal(err)
	}
	if err := ioutil.WriteFile(filepath.Join(sdir, batchTempDirName, "ab", "ab01"+batchTempSuffix), []byte("z"), d.FilePerm); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(base, "ab"), d.PathPerm); err != nil {
		t.Fatal(err)
	}
	if err := ioutil.WriteFile(filepath.Join(base, "ab", "ab01"), []byte("z"), d.FilePerm); err != nil {
		t.Fatal(err)
	}

	d2 := New(opts)
	defer d2.EraseAll()
	if d2.Has("ab01") {
		t.Fatal("orphan final block survived recovery")
	}
	if _, err := os.Stat(sdir); !os.IsNotExist(err) {
		t.Fatal("orphan staging directory not cleaned")
	}
	if len(allKeys(d2)) != 0 {
		t.Fatal("store should be empty after orphan cleanup")
	}
}

func TestConcurrentWriteAndBatchReaders(t *testing.T) {
	d := newTestDiskv(t, Options{CacheSizeMax: 1 << 20})
	if err := d.Write("k", []byte("00000000")); err != nil {
		t.Fatal(err)
	}
	const writers = 12
	var wg sync.WaitGroup
	wg.Add(writers * 2)
	for g := 0; g < writers; g++ {
		g := g
		go func() { // batch writer
			defer wg.Done()
			v := []byte(fmt.Sprintf("batch%02dx", g))
			if err := d.WriteBatch(fmt.Sprintf("wb-%d", g), []BatchItem{{Key: "k", Val: v}}); err != nil {
				t.Errorf("batch write: %s", err)
			}
		}()
		go func() { // plain Write writer
			defer wg.Done()
			v := []byte(fmt.Sprintf("plain%02dx", g))
			if err := d.Write("k", v); err != nil {
				t.Errorf("write: %s", err)
			}
		}()
	}

	stop := make(chan struct{})
	var rwg sync.WaitGroup
	rwg.Add(6)
	for i := 0; i < 6; i++ {
		go func() {
			defer rwg.Done()
			for {
				select {
				case <-stop:
					return
				default:
					got, err := d.Read("k")
					if err != nil {
						t.Errorf("read: %s", err)
						return
					}
					if len(got) != len("batch00x") {
						t.Errorf("torn read length: %q", got)
						return
					}
				}
			}
		}()
	}
	wg.Wait()
	close(stop)
	rwg.Wait()
}

func TestWriteBatchEraseConsistency(t *testing.T) {
	d := newTestDiskv(t, Options{
		Index:     &BTreeIndex{},
		IndexLess: func(a, b string) bool { return a < b },
	})
	if err := d.WriteBatch("b", []BatchItem{
		{Key: "x", Val: []byte("1")},
		{Key: "y", Val: []byte("2")},
	}); err != nil {
		t.Fatal(err)
	}
	if err := d.Erase("x"); err != nil {
		t.Fatal(err)
	}
	if d.Has("x") {
		t.Fatal("Erase ineffective after batch")
	}
	if got := d.Index.Keys("", 10); len(got) != 1 || got[0] != "y" {
		t.Fatalf("index after Erase: %v", got)
	}
	// Re-creating an erased key via another batch.
	if err := d.WriteBatch("b2", []BatchItem{{Key: "x", Val: []byte("3")}}); err != nil {
		t.Fatal(err)
	}
	got, err := d.Read("x")
	if err != nil || string(got) != "3" {
		t.Fatalf("recreate: %q %v", got, err)
	}
}

func TestWriteBatchCacheGeneration(t *testing.T) {
	d := newTestDiskv(t, Options{CacheSizeMax: 1 << 20})
	if err := d.Write("k", []byte("v1")); err != nil {
		t.Fatal(err)
	}

	// Start a slow read that captures the v1 generation, then overwrite via
	// batch before the siphon reaches EOF.
	rc, err := d.ReadStream("k", false)
	if err != nil {
		t.Fatal(err)
	}
	buf := make([]byte, 1)
	if _, err := io.ReadFull(rc, buf); err != nil {
		t.Fatal(err)
	}
	if err := d.WriteBatch("over", []BatchItem{{Key: "k", Val: []byte("v2-longer-value")}}); err != nil {
		t.Fatal(err)
	}
	rest, _ := ioutil.ReadAll(rc)
	rc.Close()
	if string(append(buf, rest...)) != "v1" {
		t.Fatal("slow read did not return the complete old value")
	}
	if d.isCached("k") {
		t.Fatal("stale siphon backfilled an old generation into the cache")
	}
	got, err := d.Read("k")
	if err != nil || string(got) != "v2-longer-value" {
		t.Fatalf("new value unavailable: %q %v", got, err)
	}
}
