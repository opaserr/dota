# Dose calculation via Transformers #

This repository contains the code to calculate the dose deposited by a mono-energetic beam of protons, for arbitrary patient geometries and beam energies. 

* GNU General Public License 3.0
* Copyright: Oscar Pastor-Serrano, TU Delft

The 'main' branch contains the code for proton beamlets, while the 'dev_photons' branch contains the code for predictin full photon beams.

### Credits ###

If you like this repository, please click on Star!

If you use the code for your research, please consider citing:

Pastor-Serrano, O., & Perkó, Z. (2021). Learning the Physics of Particle Transport via Transformers.
<https://doi.org/10.1609/aaai.v36i11.21466>

Pastor-Serrano, O., & Perkó, Z. (2022). Millisecond speed deep learning based proton dose calculation with Monte Carlo accuracy.
<https://doi.org/10.1088/1361-6560/ac692e>

Pastor-Serrano, O., Dong, P., Huang, C., Xing, L., & Perkó, Z. (2023). Sub-second photon dose prediction via transformer neural networks.
<https://doi.org/10.1002/mp.16231> 

This project is supported by the following institutions:

* KWF Kanker Bestrijding
* Department of Radiation Science and Technology (TU Delft)

### Requirements ###

* Tensorflow 2.5 or higher
* pymedphys
* tensorflow-addons
