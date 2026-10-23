# Original candidate baseline evidence

The two ZIP files are the original supplied results, copied byte-for-byte.
`manifest.json` records their SHA256 checksums and key measurement metadata.
Each ZIP contains `predictions.jsonl`, `metrics.json`, and `baseline_report.json`;
Granite additionally contains the two inference shards.

| ZIP | Pinned model revision | Correct / N |
| --- | --- | --- |
| `dev-olmo2-1b-sft.zip` | `0d85a3d037876ce6ac7d4311d994400fc66ac27f` | 101 / 544 |
| `dev-granite-3.3-2b-ift.zip` | `707f574c62054322f6b5b04b6d075f0a8f05e0f0` | 344 / 544 |

Both runs used the `exact-rational-v1` dev cohort, greedy decoding, a completion
budget of 1024, and the data-manifest SHA256
`d63a78d355a96b2dbe2ab78c27c3ac86638fd1c95685064181cd9013eb4413f7`.
The reports include each system prompt and chat-template hash. The dataset
manifest itself was not included in the supplied ZIPs; its checksum is retained
as reported, not claimed to have been independently reconstructed.

The current research cohort is `boxed-numeric-v2`. It accepts decimals and
escaped percentages and has stricter fraction syntax. The current parser must
not be used to silently overwrite these historical scores. New evaluations and
future RLOO results belong in separate run directories.
