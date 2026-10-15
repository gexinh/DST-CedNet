# Source attribution

## Code and checkpoint

The original public code was released as `gexinh/ESI-DAE`, at commit `51474e6`.
The repository is now named `gexinh/DST-CedNet`. Its Git history preserves the
earlier training and MATLAB drafts.

The maintained Python package and MATLAB functions implement data synthesis,
training, inference and evaluation. Their original contributions use the MIT
license. The MATLAB patch sampler and SVD extractor use base MATLAB operations;
they do not include the earlier Brainstorm, SPM, Stafford, SEFA or STTONNICA
helper implementations.

`checkpoints/legacy_meg.h5` is preserved byte-for-byte from the original public
repository. SHA-256:

```text
7c3b29652374b3bdeed005474a5a1e664c9c970a5400103b0060618a0f9ad963
```

Its architecture, input contract and compatibility loader are documented in
[MODEL.md](MODEL.md). The maintained training profiles are separate from this
historical model.

## Modeling assets

The simulation folder derives from the original repository's Colin27-named
simulation support directory. It retains the numerical lead field, cortical
vertices, faces, connectivity and temporal basis. Cortex history, registration,
atlas and unrelated metadata are omitted. These modeling assets retain their
source terms; the software MIT license does not relicense third-party anatomy.
For anatomical/template provenance, consult the
[Brainstorm anatomy documentation](https://neuroimage.usc.edu/brainstorm/Tutorials/DefaultAnatomy)
and [Colin27 template documentation](https://www.bic.mni.mcgill.ca/ServicesAtlases/Colin27).

User-specific recordings and head models are supplied separately. The repository
does not distribute the local real EEG/MEG source recordings.

## Paper and dependencies

The paper is published by IEEE at
[DOI 10.1109/TNNLS.2022.3209925](https://doi.org/10.1109/TNNLS.2022.3209925).
The figures in `assets/` illustrate the method and have separate terms from the
software license.

MATLAB and its Engine API are MathWorks products and require a suitable MATLAB
installation. TensorFlow, NumPy, SciPy, PyYAML, h5py and optional plotting tools
retain their own licenses.

## Comparison methods

The maintained MNE/wMNE, graph-Laplacian LORETA, SBL and FOCUSS functions are
independent implementations of the algorithms listed in
[REFERENCES.md](REFERENCES.md). The earlier SBL/FOCUSS wrappers called
`sparse_learning.m`, whose header credits David Wipf. That historical source
file is not bundled with the maintained package.

The STTONNICA adapter calls an external MATLAB package supplied by the user.
The historical package headers credit the Cuban Neuroscience Center
(2009–2016) and Joint China–Cuba Lab (2016); its accompanying algorithm-update
note cites [Valdés-Sosa et al., 2009](https://doi.org/10.1002/hbm.20784).
No license grant or software release number was identified in those supplied
files. Obtain the implementation and applicable terms from its authors. The
adapter's license applies to the adapter code; external method files retain
their own terms.
