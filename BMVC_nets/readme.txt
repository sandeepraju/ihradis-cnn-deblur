S14_19_200.deploy 
- deploy protobuf for all S14_19 networks
- net input is set to 200x200, output is 150x150

S14_19_FQ_178000.model
- L15 network from Table 1 in [1] used to get results on artifical data from Figure 3.
- Expects text at DPI 120-150 DPI and perfect black and white levels.

S14_19_real_130000.model
- L15 network from Table 1 in [1] trained on artificial data for real photo deblurring. 
This network was used to get results in Figures 4 and 5.
- Expects reasonable black and white levels. Fails on dark and overexposed images.
- Expects reasonable page orientation and resolution 120-150 DPI.

[1] Hradiš et al.: Convolutional Neural Networks for Direct Text Deblurring. BMVC 2015.
