# Drop your profile documents here

Put **plain-text or markdown** versions of the following in this directory
(the profile engine reads every `*.txt` and `*.md` except this README):

- `resume.txt` - your resume, exported from PDF as text
- `projects.md` - write-ups of your key projects (the quant swarm itself is a
  strong one: Markov-Switching GMMs, Numba CUDA Euler-Maruyama, vLLM on 4×H100,
  SLURM orchestration, SEC EDGAR NLP pipeline)
- `thesis.txt` - MS thesis abstract / summary
- anything else that shows who you are technically

**Include your GitHub URL and any personal site** somewhere in these files -
the memo synthesizer extracts them and cites your GitHub as the instant
proof-of-work link in every draft.

The profile is rebuilt automatically whenever these files change (SHA-1
watch); otherwise the cached `profile.json` + `profile.npy` are reused.
