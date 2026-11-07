<div align="center">

# Non-Code Runtime Resources

[Project](../../README.md) | [Managed setup](../setup/MANAGED_SETUP.md) | [Terms](MODEL_ASSET_TERMS.md) | [中文](EXTERNAL_RESOURCES.zh-CN.md)

</div>

---

## Scope

The [SlicerDSpace-models](https://github.com/ChengjinLii/SlicerDSpace-models)
repository covers weights, templates, atlases and required inference references.
Existing local asset directories remain usable without renaming or redownloading.
Raw subject images, analysis outputs,
training/test datasets, Conda/Python environments, CUDA libraries, executables
and subject-specific fitted dictionaries are not resource-repository contents.
The default 31-file core profile is unchanged. Optional resources remain opt-in.

| Resource | Directory | Policy |
| --- | --- | --- |
| PNL CNN weights/architecture | `brain_mask/pnl_cnn/` | Three HDF5 files, architecture and full upstream agreement |
| IIT b0 template | Same local directory | Git-ignored, upstream-only; electronic redistribution requires written IIT permission |
| TractCloud inference subset | `bundles/tractcloud/` | Model, arguments, 308-byte reference and upstream agreement; no training/test archives |
| WMA / ORG atlases | `bundles/wma/archives/` | Two pinned official ZIPs, approximately 1.68 GiB; local extracted copies are Git-ignored |
| WMQL / SAF / MK-Curve small data | Extension runtime package | Query files/direction tables stay packaged; not duplicated as large weights |

---

## Provenance

PNL uses author release v0.3; TractCloud's minimal inference subset matches the
saved v1.0.0 archives.
ORG ZIP sizes/MD5 match the fixed [Zenodo record](https://zenodo.org/records/2648292),
whose metadata states CC BY 4.0. SHA256, citations and the original Slicer/BWH
agreement are retained. The [IIT terms](https://www.nitrc.org/projects/iit/) require
written permission for electronic reposting; a private mirror does not remove
that restriction. These engineering records are not legal certification.

---

## Interfaces

`models/external_assets_manifest.json` records optional sizes, hashes, sources
and redistribution policies. It does not overwrite the core catalog.
The optional mirror has its own JSON/TSV with 14 approved-subset files, including
the pinned SHA256 index of all 849 extracted ORG files.
The resource commit `11167a5` has been pushed, including eight LFS binary objects.
The mirror URL is pinned to that commit. The code repository remains private;
the separately hosted resource repository was already public when checked.
No repository visibility was changed by this work.

```bash
PythonSlicer scripts/download_model_assets.py \
  --manifest models/external_assets_manifest.json --profile mirrorable-inference \
  --asset-root /path/to/SlicerDSpace-models --verify-only
```

`scripts/import_external_resources.py` defaults to a plan and requires
`--execute` for copying. Sources are explicit and hash-checked; originals are
retained and differing existing files are not overwritten. ORG extraction
rejects traversal/symlinks, omits macOS forks and records 849 runtime files
verified against the official ZIP bytes.

PNL, TractCloud and WMA can resolve complete, hash-verified resources from the
configured shared asset root when no backend-specific path is supplied.
Explicit paths and legacy environment variables retain precedence. WMA checks
the extracted files against the pinned index, not just the existence of ZIPs
or empty directories.

---

## Selected Preparation

In Slicer, choose algorithms and use **Environment and installation > Prepare
selected analysis environment**. Preparation requests only missing resources
for the selected routes, shows full-profile size and terms, and runs downloads,
SHA256 checks and ORG extraction in a cancellable background process. Verified
paths are saved after success, followed by a fresh environment check. A failed
or cancelled resource phase does not replace the previous configured paths.
Existing valid files are reused. A differing existing resource is not silently
overwritten: inspect it or choose a clean resource root.

PNL runtime setup reuses the verified shared model/template if available;
otherwise it obtains the author's model archive and IIT's upstream template,
then imports verified bytes into the same shared root. Python, ANTs and source
code stay in the isolated runtime. Legacy prefix-local PNL installations remain
supported. No GPU setting, algorithm parameter or scientific-QC threshold changes.

The equivalent CLI defaults to a plan; execution requires explicit confirmation:

```bash
PythonSlicer scripts/setup_optional_assets.py --profiles tractcloud wma
PythonSlicer scripts/setup_optional_assets.py --profiles tractcloud wma \
  --asset-root /path/to/resources --execute --accept-terms
PythonSlicer scripts/setup_optional_assets.py --profiles tractcloud wma \
  --asset-root /path/to/resources --from-directory /path/to/offline-resources --execute
```

For offline use, import only the selected files into the chosen root and extract
ORG archives with the CLI above. Slicer's **Use offline model resources** checks
selected core and missing optional profiles without network access. Weights do
not install Python packages or external tools. WMA/TractCloud source-runtime
blockers remain visible rather than being marked ready after downloading data.

The public resource mirror supports anonymous downloads; users do not need Git
or Git LFS for Slicer's URL downloader. Anonymous TractCloud acquisition in a
clean resource root passed all four file size/SHA256 checks. Official URLs remain
available for ORG, IIT and PNL runtime acquisition. A future private
mirror must declare `access: authorized-private`: preparation then fails before
partial downloads and directs users to an authorized checkout or offline import.

Local evidence is under `ignore/validation/external_resources_20261021/`.
Byte-preserving migration is not new validation of human scientific QC.

Checks: 14 mirrorable files, six complete local PNL files and the unchanged
31-file core passed verification. PNL passed all three GPU forward checks from
the new directory. The installed GUI saved the PNL/WMA/TractCloud resource paths
and its selected-default environment still reports zero blockers. Python
regression: 1,287 tests, 1,284 passed and three skipped. Independent Slicer
environment/setup GUI tests and package resource audit passed. A second GUI
test exercised the actual optional-resource worker and automatic path saving
without network access. A real loopback HTTP download test verified first
acquisition and network-free reuse. Anonymous mirror acquisition also passed;
the independent full remote readback passed all 14 sizes/SHA256 values from a
fresh GitHub checkout of `11167a5`, with no LFS pointers left in selected files.
Evidence: `ignore/validation/external_resources_20261021/remote-verified.lock.json`.

Network access is still a prerequisite: a public URL is not a guarantee that a
firewall or isolated container can reach it. The downloader respects standard
proxy environment settings and retains resumable partial files. Timeouts remain
failures, not ready states; offline import is the network-independent option.
An isolated bridge-network attempt with Slicer's Python timed out and was retained
as a failure. The same Slicer 5.10 Python worker passed anonymous acquisition of
all four TractCloud files using a reachable host network and the existing standard
proxy environment, without disabling TLS verification. No proxy address is coded
into the extension. Evidence: `ignore/validation/slicer_anonymous_resources_20261021/`.
