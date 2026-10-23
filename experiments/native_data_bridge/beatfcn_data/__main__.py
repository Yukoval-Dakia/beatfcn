import argparse
import json
from pathlib import Path
from .bridge import build_all, load_artifact, file_sha, require

def main():
    p = argparse.ArgumentParser(description='Offline native osu4K supervision bridge. No training/upload support.')
    sub = p.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build'); b.add_argument('--source', type=Path, required=True)
    b.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1])
    v = sub.add_parser('validate'); v.add_argument('--source-root', type=Path); v.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = p.parse_args()
    if args.command == 'build':
        print(json.dumps(build_all(args.source, args.output), indent=2))
    else:
        manifest = json.loads((args.root / 'manifest.json').read_text())
        total = 0
        for e in manifest['entries']:
            path = args.root / e['path']
            require(file_sha(path) == e['sha256'], 'artifact integrity mismatch')
            load_artifact(path, verify_sources=True, source_root=args.source_root); total += 1
        require(len({e['song_group'] for e in manifest['entries']}) == total, 'duplicate training song')
        print(json.dumps({'validated': total, 'unsupported': manifest['unsupported'],
                          'source_hashes_verified': True, 'train_only': True}, indent=2))

if __name__ == '__main__': main()
