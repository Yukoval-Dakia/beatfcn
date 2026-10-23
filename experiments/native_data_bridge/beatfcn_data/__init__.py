"""Local native osu!mania label bridge; no model, training, or upload API."""
from .bridge import (DataError, Unsupported, parse_osu, build_artifact, validate_artifact,
                     load_artifact, model_condition, teacher_forcing_steps,
                     verify_source_files, build_all, endpoint_teacher_forcing_fields)
