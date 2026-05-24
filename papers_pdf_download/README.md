# Bundled example papers

These five PDFs ship with DeepRead so the demo works offline, without
any network roundtrip the first time a user clicks a paper.

| slug | title | year | pages | source |
|---|---|---:|---:|---|
| `attention.pdf` | Attention Is All You Need | 2017 | 15 | [arXiv:1706.03762](https://arxiv.org/pdf/1706.03762) |
| `gfs.pdf` | The Google File System | 2003 | 15 | [research.google.com](https://static.googleusercontent.com/media/research.google.com/en//archive/gfs-sosp2003.pdf) |
| `mapreduce.pdf` | MapReduce | 2004 | 13 | [research.google.com](https://static.googleusercontent.com/media/research.google.com/en//archive/mapreduce-osdi04.pdf) |
| `raft.pdf` | In Search of an Understandable Consensus Algorithm | 2014 | 18 | [raft.github.io](https://raft.github.io/raft.pdf) |
| `bitcoin.pdf` | Bitcoin: A Peer-to-Peer Electronic Cash System | 2008 | 9 | [bitcoin.org](https://bitcoin.org/bitcoin.pdf) |

Total bundle size: ~3.3 MB.

## Why bundled

DeepRead's whole premise is *local-first*. A demo that requires a
network round-trip to arXiv or bitcoin.org the first time you click
something contradicts that promise (and breaks the moment the user is
offline, or arXiv times out, or the URL 302s somewhere new).

Bundling the papers means:
- Zero network calls on the hot path
- "0 bytes sent to cloud" is literally true at runtime
- The demo works on a flight

## Refreshing

If you need to re-download from canonical URLs (e.g. after a refresh
of upstream content):

```bash
uv run python scripts/refresh_papers.py
```

This is the only path in the codebase that talks to the network.
