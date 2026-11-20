// Atomic, index-coupled batch writes for Diskv.
//
// A batch write follows a two-phase protocol:
//
//   - prepare phase: every value is written to a recoverable temporary file
//     under a per-batch staging directory and synced to disk. No final data
//     file and no index entry is touched during this phase, so a failure leaves
//     the store exactly as it was before the batch.
//   - commit phase: existing final files are moved into a backup area inside
//     the staging directory, the temporary files are atomically renamed into
//     their final locations, and only then are the in-memory cache, the
//     generation counters and the index updated. A checkpoint and a commit
//     marker make the batch recognizable to a later recovery run.
//
// Staging layout (root is TempDir when configured, otherwise BasePath):
//
//	<root>/.diskv-batch-staging/
//	    .committed/<batchID>     durable markers of completed batch IDs
//	    <batchID>/
//	        checkpoint.json       recognizable batch descriptor
//	        committed             commit marker (present == decided to commit)
//	        tmp/<rel-path>.tmp    synced temporary data blocks
//	        backup/<rel-path>     previous final files captured on overwrite
//
// On reopen, recoverBatches scans the staging root: batches without a commit
// marker are rolled back (backups restored, installed new files removed),
// batches with a marker are rolled forward (missing finals installed, leftovers
// removed). Completed batch IDs are retained as marker files in .committed, so
// a re-submitted batch ID stays a no-op across process restarts. Either way,
// recovery never needs the checkpoint to locate files: the tmp/ and backup/
// trees mirror the final layout, so orphans are cleaned up even when the
// checkpoint itself was never written.

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
	"strings"
)

const (
	batchStagingDirName  = ".diskv-batch-staging"
	batchRegistryDirName = ".committed"
	batchCheckpointName  = "checkpoint.json"
	batchMarkerName      = "committed"
	batchTempDirName     = "tmp"
	batchBackupDirName   = "backup"
	batchTempSuffix      = ".tmp"
)

// Batch item and error types.

// BatchItem is a single key-value pair handed to WriteBatch.
type BatchItem struct {
	Key string
	Val []byte
}

// BatchStreamItem is a single key-reader pair handed to WriteBatchStream.
type BatchStreamItem struct {
	Key    string
	Reader io.Reader
}

// BatchItemError records why a single key failed within a batch.
type BatchItemError struct {
	Key string
	Err error
}

// BatchError is returned when a batch cannot be committed. Every failed key is
// listed with its reason. When WriteBatchStream returns a *BatchError, all
// temporary files of the batch have been removed and every pre-existing index
// entry and data file is unchanged.
type BatchError struct {
	BatchID string
	Errors  []BatchItemError
}

// Error implements the error interface.
func (e *BatchError) Error() string {
	parts := make([]string, 0, len(e.Errors))
	for _, ie := range e.Errors {
		parts = append(parts, fmt.Sprintf("%s: %s", ie.Key, ie.Err))
	}
	return fmt.Sprintf("batch %s failed: %s", e.BatchID, strings.Join(parts, "; "))
}

var (
	errEmptyBatch   = errors.New("empty batch")
	errEmptyBatchID = errors.New("empty batch ID")
	errBadBatchID   = errors.New("bad batch ID")
	errDuplicateKey = errors.New("duplicate key in batch")
)

// batchItemError pairs a failing key with its cause during the commit phase.
type batchItemError struct {
	key string
	err error
}

// batchCheckpoint is the on-disk, recognizable descriptor of a batch.
type batchCheckpoint struct {
	BatchID string            `json:"batch_id"`
	Items   []batchItemRecord `json:"items"`
}

type batchItemRecord struct {
	Key    string `json:"key"`
	Rel    string `json:"rel"`
	Backup bool   `json:"backup"`
}

// WriteBatch writes all key-value pairs as one atomic batch. The call either
// commits every pair or none of them: the in-memory and disk indices are
// updated together, and only after every data block has been synced.
//
// batchID identifies the batch for idempotency: re-submitting a batch ID that
// has already committed is a no-op and never applies the second payload a
// second time.
func (d *Diskv) WriteBatch(batchID string, items []BatchItem) error {
	streamItems := make([]BatchStreamItem, len(items))
	for i, it := range items {
		streamItems[i] = BatchStreamItem{Key: it.Key, Reader: bytes.NewReader(it.Val)}
	}
	return d.WriteBatchStream(batchID, streamItems)
}

// WriteBatchString is a convenience wrapper around WriteBatch for string
// values.
func (d *Diskv) WriteBatchString(batchID string, items map[string]string) error {
	batchItems := make([]BatchItem, 0, len(items))
	for k, v := range items {
		batchItems = append(batchItems, BatchItem{Key: k, Val: []byte(v)})
	}
	return d.WriteBatch(batchID, batchItems)
}

// WriteBatchStream is the streaming entry point for an atomic batch write.
// Readers are consumed fully and their blocks synced before any index update.
// On failure a *BatchError is returned describing the offending key and the
// underlying reason, and the store keeps its previous state.
func (d *Diskv) WriteBatchStream(batchID string, items []BatchStreamItem) error {
	if len(items) == 0 {
		return errEmptyBatch
	}
	if err := validateBatchID(batchID); err != nil {
		return err
	}

	// Validate every key up front, before any file or lock is touched.
	records := make([]batchItemRecord, 0, len(items))
	seenKeys := map[string]struct{}{}
	seenRels := map[string]struct{}{}
	for _, it := range items {
		if it.Key == "" {
			return &BatchError{BatchID: batchID, Errors: []BatchItemError{{Key: it.Key, Err: errEmptyKey}}}
		}
		pathKey := d.transform(it.Key)
		if err := validatePathKey(pathKey); err != nil {
			return &BatchError{BatchID: batchID, Errors: []BatchItemError{{Key: it.Key, Err: err}}}
		}
		rel := relPathFor(pathKey)
		if _, ok := seenKeys[it.Key]; ok {
			return &BatchError{BatchID: batchID, Errors: []BatchItemError{{Key: it.Key, Err: errDuplicateKey}}}
		}
		if _, ok := seenRels[rel]; ok {
			return &BatchError{BatchID: batchID, Errors: []BatchItemError{{Key: it.Key, Err: errDuplicateKey}}}
		}
		seenKeys[it.Key] = struct{}{}
		seenRels[rel] = struct{}{}
		records = append(records, batchItemRecord{Key: it.Key, Rel: rel})
	}

	d.mu.Lock()
	defer d.mu.Unlock()
	return d.writeBatchStreamWithLock(batchID, items, records)
}

// writeBatchStreamWithLock runs the two-phase protocol while holding d.mu.
func (d *Diskv) writeBatchStreamWithLock(batchID string, items []BatchStreamItem, records []batchItemRecord) error {
	sdir := d.stagingDirWithLock(batchID)
	markerPath := filepath.Join(sdir, batchMarkerName)

	// Idempotency: an already completed batch ID must not re-apply its
	// payload, even if its staging cleanup is still pending on disk.
	if _, ok := d.completedBatches[batchID]; ok {
		return nil
	}
	if _, err := os.Stat(markerPath); err == nil {
		d.forwardBatchWithLock(sdir)
		_ = d.recordCommittedBatchWithLock(batchID)
		os.RemoveAll(sdir)
		d.completedBatches[batchID] = struct{}{}
		return nil
	}
	if _, err := os.Stat(sdir); err == nil {
		// Leftovers from a run that never reached the commit decision:
		// no final file was logically claimed, so discard them.
		d.rollbackBatchWithLock(sdir)
	}

	// ---- prepare phase: write and sync every block in staging ----
	tmpRoot := filepath.Join(sdir, batchTempDirName)
	if err := os.MkdirAll(tmpRoot, d.PathPerm); err != nil {
		return fmt.Errorf("batch staging mkdir: %s", err)
	}

	for i, it := range items {
		tmpPath := filepath.Join(tmpRoot, records[i].Rel+batchTempSuffix)
		if err := d.writeTempBlockWithLock(tmpPath, it.Reader); err != nil {
			os.RemoveAll(sdir) // best effort; nothing outside staging exists yet
			return &BatchError{
				BatchID: batchID,
				Errors:  []BatchItemError{{Key: it.Key, Err: err}},
			}
		}
	}

	// ---- commit phase ----
	if err := d.commitBatchWithLock(batchID, sdir, records); err != nil {
		return &BatchError{
			BatchID: batchID,
			Errors:  []BatchItemError{{Key: err.key, Err: err.err}},
		}
	}

	// Retain a recognizable, durable record of the completed batch ID so a
	// duplicate submission after a restart cannot re-apply the payload. Only
	// then is the staging directory reclaimed.
	if err := d.recordCommittedBatchWithLock(batchID); err != nil {
		// The data is committed; losing only the idempotency record is not a
		// data failure, and recovery will re-create it from the commit marker.
		_ = err
	}
	os.RemoveAll(sdir)
	d.completedBatches[batchID] = struct{}{}
	return nil
}

// commitBatchWithLock installs all blocks, then updates cache, generations and
// index as one critical section. Any failure triggers a full rollback, so the
// pre-batch index entries remain valid and visible throughout.
func (d *Diskv) commitBatchWithLock(batchID, sdir string, records []batchItemRecord) *batchItemError {
	backupRoot := filepath.Join(sdir, batchBackupDirName)

	// Capture every final file that is about to be overwritten.
	for i := range records {
		finalPath := d.finalPathForRel(records[i].Rel)
		if _, err := os.Stat(finalPath); err != nil {
			continue // new key, nothing to back up
		}
		backupPath := filepath.Join(backupRoot, records[i].Rel)
		if err := os.MkdirAll(filepath.Dir(backupPath), d.PathPerm); err != nil {
			return d.abortBatchWithLock(sdir, records, records[i].Key, fmt.Errorf("backup mkdir: %s", err))
		}
		if err := os.Rename(finalPath, backupPath); err != nil {
			return d.abortBatchWithLock(sdir, records, records[i].Key, fmt.Errorf("backup: %s", err))
		}
		records[i].Backup = true
	}

	// Make the batch recognizable before the final renames.
	if err := d.writeCheckpointWithLock(sdir, batchID, records); err != nil {
		return d.abortBatchWithLock(sdir, records, records[0].Key, err)
	}

	// Atomically install every block.
	for i := range records {
		tmpPath := filepath.Join(sdir, batchTempDirName, records[i].Rel+batchTempSuffix)
		finalPath := d.finalPathForRel(records[i].Rel)
		if err := os.MkdirAll(filepath.Dir(finalPath), d.PathPerm); err != nil {
			return d.abortBatchWithLock(sdir, records, records[i].Key, fmt.Errorf("final mkdir: %s", err))
		}
		if err := os.Rename(tmpPath, finalPath); err != nil {
			return d.abortBatchWithLock(sdir, records, records[i].Key, fmt.Errorf("rename: %s", err))
		}
		if err := fsyncDir(filepath.Dir(finalPath)); err != nil {
			return d.abortBatchWithLock(sdir, records, records[i].Key, fmt.Errorf("final dir sync: %s", err))
		}
	}

	// Durably publish the commit decision while memory and index still show
	// the old state. A failure here can cleanly roll back because no index
	// entry has been touched.
	if err := writeSyncedFile(filepath.Join(sdir, batchMarkerName), nil, d.FilePerm, d.PathPerm); err != nil {
		return d.abortBatchWithLock(sdir, records, records[0].Key, fmt.Errorf("commit marker: %s", err))
	}

	// Every block is durably in place and the decision is committed: now the
	// in-memory cache, generations and index switch to the new generation as
	// one critical section. Readers holding RLock either saw the complete old
	// value or now see the complete new value.
	for _, rec := range records {
		if d.Index != nil {
			d.Index.Insert(rec.Key)
		}
		d.bustCacheWithLock(rec.Key)
		d.touchGenerationWithLock(rec.Key)
	}

	return nil
}

// abortBatchWithLock restores every backup, removes every newly installed
// final file, and tears down the staging directory. The index has not been
// touched at this point, so pre-existing entries stay intact.
func (d *Diskv) abortBatchWithLock(sdir string, records []batchItemRecord, key string, cause error) *batchItemError {
	for _, rec := range records {
		finalPath := d.finalPathForRel(rec.Rel)
		if rec.Backup {
			backupPath := filepath.Join(sdir, batchBackupDirName, rec.Rel)
			if _, err := os.Stat(backupPath); err == nil {
				if err := os.Rename(backupPath, finalPath); err != nil {
					return &batchItemError{key: rec.Key, err: fmt.Errorf("rollback restore: %s (original error: %s)", err, cause)}
				}
				fsyncDir(filepath.Dir(finalPath)) // best effort
			}
		} else {
			if err := os.Remove(finalPath); err != nil && !os.IsNotExist(err) {
				return &batchItemError{key: rec.Key, err: fmt.Errorf("rollback remove: %s (original error: %s)", err, cause)}
			}
		}
	}
	os.RemoveAll(sdir)
	for _, rec := range records {
		d.pruneDirsWithLock(rec.Key)
	}
	return &batchItemError{key: key, err: cause}
}

// writeTempBlockWithLock streams one value into a synced temporary block,
// transparently applying the configured compression.
func (d *Diskv) writeTempBlockWithLock(tmpPath string, r io.Reader) error {
	if err := os.MkdirAll(filepath.Dir(tmpPath), d.PathPerm); err != nil {
		return fmt.Errorf("temp mkdir: %s", err)
	}

	f, err := os.OpenFile(tmpPath, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, d.FilePerm)
	if err != nil {
		return fmt.Errorf("temp file: %s", err)
	}

	wc := io.WriteCloser(&nopWriteCloser{f})
	if d.Compression != nil {
		wc, err = d.Compression.Writer(f)
		if err != nil {
			f.Close()
			os.Remove(f.Name())
			return fmt.Errorf("compression writer: %s", err)
		}
	}

	if _, err := io.Copy(wc, r); err != nil {
		f.Close()
		os.Remove(f.Name())
		return fmt.Errorf("i/o copy: %s", err)
	}
	if err := wc.Close(); err != nil {
		f.Close()
		os.Remove(f.Name())
		return fmt.Errorf("compression close: %s", err)
	}
	if err := f.Sync(); err != nil {
		f.Close()
		os.Remove(f.Name())
		return fmt.Errorf("file sync: %s", err)
	}
	if err := f.Close(); err != nil {
		return fmt.Errorf("file close: %s", err)
	}
	return fsyncDir(filepath.Dir(tmpPath))
}

// writeCheckpointWithLock persists the recognizable batch descriptor.
func (d *Diskv) writeCheckpointWithLock(sdir, batchID string, records []batchItemRecord) error {
	cp := batchCheckpoint{BatchID: batchID, Items: records}
	data, err := json.MarshalIndent(&cp, "", "  ")
	if err != nil {
		return fmt.Errorf("checkpoint encode: %s", err)
	}
	if err := writeSyncedFile(filepath.Join(sdir, batchCheckpointName), data, d.FilePerm, d.PathPerm); err != nil {
		return fmt.Errorf("checkpoint: %s", err)
	}
	return nil
}

// ---- recovery ----

// recoverBatches resolves every staging directory left behind by an interrupted
// process. It runs from New, before the index is initialized, so the index
// rebuild sees exactly the files recovery leaves behind.
func (d *Diskv) recoverBatches() error {
	// Load the durable idempotency record first.
	d.completedBatches = map[string]struct{}{}
	if ids, err := d.loadCommittedBatches(); err == nil {
		for _, id := range ids {
			d.completedBatches[id] = struct{}{}
		}
	}

	stagingRoot := d.stagingRoot()
	entries, err := ioutil.ReadDir(stagingRoot)
	if os.IsNotExist(err) {
		return nil
	}
	if err != nil {
		return err
	}

	for _, entry := range entries {
		if !entry.IsDir() || entry.Name() == batchRegistryDirName {
			continue
		}
		batchID := entry.Name()
		sdir := filepath.Join(stagingRoot, batchID)
		if _, err := os.Stat(filepath.Join(sdir, batchMarkerName)); err == nil {
			d.forwardBatchWithLock(sdir)
			if err := d.recordCommittedBatchWithLock(batchID); err != nil {
				_ = err
			}
			d.completedBatches[batchID] = struct{}{}
		} else {
			d.rollbackBatchWithLock(sdir)
		}
		os.RemoveAll(sdir)
	}
	return nil
}

// stagingRoot is the directory containing per-batch staging directories and the
// committed-batch registry.
func (d *Diskv) stagingRoot() string {
	return filepath.Join(d.batchRoot(), batchStagingDirName)
}

// committedRegistryPath returns the marker file recording a completed batch.
func (d *Diskv) committedRegistryPath(batchID string) string {
	return filepath.Join(d.stagingRoot(), batchRegistryDirName, batchID)
}

// recordCommittedBatchWithLock durably records a completed batch ID.
func (d *Diskv) recordCommittedBatchWithLock(batchID string) error {
	return writeSyncedFile(d.committedRegistryPath(batchID), nil, d.FilePerm, d.PathPerm)
}

// loadCommittedBatches lists every batch ID whose completion was recorded.
func (d *Diskv) loadCommittedBatches() ([]string, error) {
	entries, err := ioutil.ReadDir(filepath.Join(d.stagingRoot(), batchRegistryDirName))
	if os.IsNotExist(err) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	ids := make([]string, 0, len(entries))
	for _, entry := range entries {
		if !entry.IsDir() {
			ids = append(ids, entry.Name())
		}
	}
	return ids, nil
}

// rollbackBatchWithLock undoes a batch that never published its commit marker:
// backups are restored over any newly installed file, and blocks for new keys
// are removed. The backup/ and tmp/ trees mirror the final layout, so this works
// without a readable checkpoint.
func (d *Diskv) rollbackBatchWithLock(sdir string) {
	backupRoot := filepath.Join(sdir, batchBackupDirName)
	tmpRoot := filepath.Join(sdir, batchTempDirName)

	// Record which rels had a captured previous version before moving
	// anything: the restore below removes the files from backupRoot.
	hadBackup := map[string]bool{}
	filepath.Walk(backupRoot, func(path string, info os.FileInfo, err error) error {
		if err != nil || info.IsDir() {
			return nil
		}
		rel, err := filepath.Rel(backupRoot, path)
		if err != nil {
			return nil
		}
		hadBackup[rel] = true
		return nil
	})

	// 1. Restore every captured previous version.
	filepath.Walk(backupRoot, func(path string, info os.FileInfo, err error) error {
		if err != nil || info.IsDir() {
			return nil
		}
		rel, err := filepath.Rel(backupRoot, path)
		if err != nil {
			return nil
		}
		finalPath := d.finalPathForRel(rel)
		if err := os.MkdirAll(filepath.Dir(finalPath), d.PathPerm); err != nil {
			return nil
		}
		if err := os.Rename(path, finalPath); err == nil {
			fsyncDir(filepath.Dir(finalPath))
		}
		return nil
	})

	// 2. Remove finals installed for keys that had no previous version.
	filepath.Walk(tmpRoot, func(path string, info os.FileInfo, err error) error {
		if err != nil || info.IsDir() {
			return nil
		}
		rel, err := filepath.Rel(tmpRoot, path)
		if err != nil || !strings.HasSuffix(rel, batchTempSuffix) {
			return nil
		}
		rel = strings.TrimSuffix(rel, batchTempSuffix)
		if hadBackup[rel] {
			return nil // old version restored above
		}
		finalPath := d.finalPathForRel(rel)
		if err := os.Remove(finalPath); err != nil && !os.IsNotExist(err) {
			return nil
		}
		fsyncDir(filepath.Dir(finalPath))
		return nil
	})

	os.RemoveAll(sdir)
}

// forwardBatchWithLock completes a batch that published its commit marker:
// missing finals are installed from their tmp blocks, and staging is removed.
func (d *Diskv) forwardBatchWithLock(sdir string) {
	tmpRoot := filepath.Join(sdir, batchTempDirName)
	backupRoot := filepath.Join(sdir, batchBackupDirName)

	filepath.Walk(tmpRoot, func(path string, info os.FileInfo, err error) error {
		if err != nil || info.IsDir() {
			return nil
		}
		rel, err := filepath.Rel(tmpRoot, path)
		if err != nil || !strings.HasSuffix(rel, batchTempSuffix) {
			return nil
		}
		rel = strings.TrimSuffix(rel, batchTempSuffix)
		finalPath := d.finalPathForRel(rel)
		if _, err := os.Stat(finalPath); err == nil {
			return nil // already installed
		}
		if err := os.MkdirAll(filepath.Dir(finalPath), d.PathPerm); err != nil {
			return nil
		}
		if err := os.Rename(path, finalPath); err == nil {
			fsyncDir(filepath.Dir(finalPath))
			return nil
		}
		// Defensive: prefer the previous version over a missing key.
		backupPath := filepath.Join(backupRoot, rel)
		if _, err := os.Stat(backupPath); err == nil {
			os.Rename(backupPath, finalPath)
		}
		return nil
	})

	os.RemoveAll(sdir)
}

// ---- path and validation helpers ----

// batchRoot returns the directory that hosts the batch staging tree. TempDir
// is on the same device as BasePath by contract, so renames stay atomic.
func (d *Diskv) batchRoot() string {
	if d.TempDir != "" {
		return d.TempDir
	}
	return d.BasePath
}

func (d *Diskv) stagingDirWithLock(batchID string) string {
	return filepath.Join(d.stagingRoot(), batchID)
}

func (d *Diskv) finalPathForRel(rel string) string {
	return filepath.Join(d.BasePath, rel)
}

func relPathFor(pathKey *PathKey) string {
	parts := make([]string, 0, len(pathKey.Path)+1)
	parts = append(parts, pathKey.Path...)
	parts = append(parts, pathKey.FileName)
	return filepath.Join(parts...)
}

// firstPathSegment returns the first element of a filepath.Join-style relative
// path, or the empty string for empty or "." paths.
func firstPathSegment(rel string) string {
	rel = filepath.ToSlash(rel)
	if rel == "" || rel == "." {
		return ""
	}
	if i := strings.IndexByte(rel, '/'); i >= 0 {
		return rel[:i]
	}
	return rel
}

func validatePathKey(pathKey *PathKey) error {
	for _, pathPart := range pathKey.Path {
		if strings.ContainsRune(pathPart, os.PathSeparator) {
			return errBadKey
		}
	}
	if strings.ContainsRune(pathKey.FileName, os.PathSeparator) {
		return errBadKey
	}
	return nil
}

func validateBatchID(batchID string) error {
	if batchID == "" {
		return errEmptyBatchID
	}
	if batchID == "." || batchID == ".." {
		return errBadBatchID
	}
	if strings.ContainsRune(batchID, os.PathSeparator) || strings.ContainsRune(batchID, '/') || strings.ContainsRune(batchID, '\\') {
		return errBadBatchID
	}
	return nil
}

// writeSyncedFile creates or truncates path, writes data, syncs the file and
// its parent directory.
func writeSyncedFile(path string, data []byte, filePerm, pathPerm os.FileMode) error {
	if err := os.MkdirAll(filepath.Dir(path), pathPerm); err != nil {
		return err
	}
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, filePerm)
	if err != nil {
		return err
	}
	if _, err := f.Write(data); err != nil {
		f.Close()
		return err
	}
	if err := f.Sync(); err != nil {
		f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	return fsyncDir(filepath.Dir(path))
}

// fsyncDir flushes directory entry changes (renames, creates, unlinks).
func fsyncDir(dir string) error {
	f, err := os.Open(dir)
	if err != nil {
		return err
	}
	defer f.Close()
	return f.Sync()
}
