# Shared Pupu24 + KDA4 S0a experiment

Start with [HANDOFF.md](HANDOFF.md). This isolated experiment does not replace the repository's older models or training scripts.

It implements a 476,739,917-parameter numerical scaffold with 73,598,285 trainable parameters. Audio and event tokens share all 24 pretrained Pupu layers, with four new full KDA blocks. It is not a trained chart writer, an SFT result, or proof of GPU/4 GB feasibility.

- [Architecture and visibility rules](ARCHITECTURE.md)
- [Target event contract and explicit gaps](contracts/EVENT_CONTRACT.md)
- [Verification scope](VERIFICATION.md)
- [Required private inputs](contracts/modal_teacher_provenance.json) and [Higher metadata](contracts/modal_probe_candidate.json)

No audio, pretrained weights, credentials, private account records, or external execution results are included. CUDA/Modal execution has not run.
