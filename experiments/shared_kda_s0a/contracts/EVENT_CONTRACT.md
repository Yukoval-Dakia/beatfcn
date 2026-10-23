# Target task contract; decoder NOT implemented by S0a

The accompanying request schema and examples describe the intended interface, not a completed semantic validator. Historical separate event-model configurations are intentionally excluded.

Canonical storage keeps event_id, onset_us, end_us, type, location, native_ref,
exact native beat rational, offset, format version and immutable sidecar.
mania4k location is lane0..3; Phira location is line_id/side/native_x.
The model view quantizes hidden free-time targets to1ms BEFORE grouping;
observed events/sidecars are never quantized or overwritten. Exact coincident
known anchors use COPY_TIME. Invalid quantization cases are quarantined.

Outer unknown-count groups: END_GAP or NEW_GROUP, coarse10ms + fine0..9ms,
or COPY_TIME. New groups strictly increase in time. Group feedback uses only
finished group attributes, not GRU hidden state. Each nonempty group uses
CONTINUE/END_GROUP -> type -> location -> atomic hold end, with legality masks.
Hold end: fixed2s block pointer -> reveal that block's audio -> coarse10ms ->
fine1ms. Ground-truth endpoint detail must NOT enter the block predictor.

O is observed/immutable, R is explicitly paired reference, I is isolated intent.
complete/empty disallow additions; editable blank is unknown, not negative.
Only complete reference coverage licenses unselected examples. Generated groups
must never be promoted to human reference evidence. Return additions only,
region statuses, reasons and reproduction identity. Whole holds respect bounds,
permission spans and future locked occupancy. Nothing here permits rewrites.

S0a input features are synthetic eight-dimensional attributes for numerical
path tests. They are NOT a canonical event encoder, completed-group pooling,
reference evidence builder, controls vector, mode-aware location head, or
semantic request validator. Outer action/coarse/fine heads are probes only.
No GRU, COPY_TIME, atomic endpoint pointer/reveal, Phira distribution, constraints,
request converter, real CF pairing/loss, or playable chart writer is implemented.
Those are explicit S0b/P1-P2 work, not hidden inside S0a labels.
