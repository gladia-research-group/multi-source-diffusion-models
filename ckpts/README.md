## Pretrained checkpoints
It is possible to download the models we used for our paper using the following links:

 * [MSDM](https://drive.google.com/file/d/1mfozibogvNrUaeS283OBz26MWNeZv-OO/view?usp=share_link): checkpoints for generation, inpainting, and (optionally) separation.
 * [Weakly MSDM](https://drive.google.com/file/d/1A33CjKPfmaqsgSXyDqr1Pbd39y_KHgU1/view?usp=share_link): checkpoints useful only for separation.

After the download extract the tar files into this directory:
```bash
# Move into this directory
cd ckpts/

# Extract MSDM checkpoint
tar -xvf msdm.tar

# Extract Weakly MSDM checkpoint
tar -xvf weakly-msdm.tar

# Finally, you can remove the tar files
rm msdm.tar weakly-msdm.tar
```