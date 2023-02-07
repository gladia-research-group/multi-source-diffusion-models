# Multi Source Diffusion Models

 * **Paper:** ***Multi-Source Diffusion Models for Simultaneous Audio Generation and Separation*** \([arXiv.org](https://arxiv.org/abs/2302.02257)\)  <br/><br/>
 * **Authors:** *Giorgio Mariani\*, Irene Tallini\*, Emilian Postolache\*, Michele Mancusi\*, Luca Cosmo, Emanuele Rodolà*  <br/><br/>
 * **Abstract:** In this work, we define a diffusion-based generative model capable of both music synthesis and source separation by learning the score of the joint probability density of sources sharing a context. Alongside the classic total inference tasks (i.e. generating a mixture, separating the sources), we also introduce and experiment on the partial inference task of source imputation, where we generate a subset of the sources given the others (e.g., play a piano track that goes well with the drums). Additionally, we introduce a novel inference method for the separation task. We train our model on Slakh2100, a standard dataset for musical source separation, provide qualitative results in the generation settings, and showcase competitive quantitative results in the separation setting. Our method is the first example of a single model that can handle both generation and separation tasks, thus representing a step toward general audio models.
Below are some examples of the tasks we can perform with our approach.

## Generation
Here we ask the neural model to randomly generate some new music with just piano and drums:

| Sample #1    | Sample #2    |
| :----------: | :----------: |
|<audio controls><source src="media/generation/sample-1.mp3"></audio> | <audio controls><source src="media/generation/sample-2.mp3"></audio> |

| Sample #3    | Sample #4    |
| :----------: | :----------: |
| <audio controls><source src="media/generation/sample-4.mp3"></audio>| <audio controls><source src="media/generation/sample-5.mp3"></audio>|

| Sample #5    | Sample #6    |
| :----------: | :----------: |
| <audio controls><source src="media/generation/sample-3.mp3"></audio> |   <audio controls><source src="media/generation/sample-6.mp3"></audio>|


<br/><br/>

## Source Imputation (a.k.a. partial generation)
Given a drum track as input, the neural model generates the accompanying piano from scratch:

**Input Drums Track 1**
<audio controls preload="none"><source src="media/inpainting/original-2.mp3" type="audio/mpeg"> Your browser does not support the audio element.</audio>

| Sampled Piano #1 | Sampled Piano #2 |
| :----------: | :----------: |
|<audio controls preload="none"><source src="media/inpainting/sample-2-1.mp3" type="audio/mpeg"> Your browser does not support the audio element.</audio> |   <audio controls preload="none"><source src="media/inpainting/sample-2-3.mp3" type="audio/mpeg"> Your browser does not support the audio element.</audio>|

<br/><br/>

**Input Drums Track 2**
<audio controls preload="none"><source src="media/inpainting/original-1.mp3" type="audio/mpeg"> Your browser does not support the audio element.</audio>

| Sampled Piano  #1 | Sampled Piano  #2 |
| :----------: | :----------: |
|<audio controls preload="none"><source src="media/inpainting/sample-1-1.mp3" type="audio/mpeg"> Your browser does not support the audio element.</audio> |    <audio controls preload="none"><source src="media/inpainting/sample-1-2.mp3" type="audio/mpeg"> Your browser does not support the audio element.</audio> |

<br/><br/>

Similarly, given a piano track as input, the neural model is able to generate the accompanying drums:

**Input Piano Track 1**
<audio controls preload="none"><source src="media/inpainting/original-3.mp3" type="audio/mpeg"> Your browser does not support the audio element.</audio>

| Sampled Drums  #1 | Sampled Drums #2 |
| :----------: | :----------: |
| <audio controls preload="none"><source src="media/inpainting/sample-3-2.mp3" type="audio/mpeg"> Your browser does not support the audio element.</audio> |    <audio controls preload="none"><source src="media/inpainting/sample-3-3.mp3" type="audio/mpeg"> Your browser does not support the audio element.</audio>|

<br/><br/>

**Input Piano Track 2**
<audio controls preload="none"><source src="media/inpainting/original-4.mp3" type="audio/mpeg"> Your browser does not support the audio element.</audio>

| Sampled Drums #1 | Sampled Drums #2 |
| :----------: | :----------: |
|    <audio controls preload="none"><source src="media/inpainting/sample-4-3.mp3" type="audio/mpeg"> Your browser does not support the audio element.</audio> |    <audio controls preload="none"><source src="media/inpainting/sample-4-2.mp3" type="audio/mpeg"> Your browser does not support the audio element.</audio>|

<br/><br/>

## Source Separation
Finally, it is possible to use our model to extract single sources from an input mixture:

**Input Mixture 1**
<audio controls preload="none"><source src="media/separation/1/mix.mp3" type="audio/mpeg"> Your browser does not support the audio element.</audio>

| Separated Bass | Separated Drums |
| :----------: | :----------: |
|<audio controls preload="none"><source src="media/separation/1/sep-bass.mp3" type="audio/mpeg"> Your browser does not support the audio element.</audio> |    <audio controls preload="none"><source src="media/separation/1/sep-drums.mp3" type="audio/mpeg"> Your browser does not support the audio element.</audio> |

|Separated Guitar| Separated Piano|
| :----------: | :----------: |
|<audio controls preload="none"><source src="media/separation/1/sep-guitar.mp3" type="audio/mpeg"> Your browser does not support the audio element.</audio> |    <audio controls preload="none"><source src="media/separation/1/sep-piano.mp3" type="audio/mpeg"> Your browser does not support the audio element.</audio>|

<br/><br/>

**Input Mixture 2**
<audio controls preload="none"><source src="media/separation/2/mix.mp3" type="audio/mpeg"> Your browser does not support the audio element.</audio>

| Separated Bass | Separated Drums |
| :----------: | :----------: |
|<audio controls preload="none"><source src="media/separation/2/sep-bass.mp3" type="audio/mpeg"> Your browser does not support the audio element.</audio> |    <audio controls preload="none"><source src="media/separation/2/sep-drums.mp3" type="audio/mpeg"> Your browser does not support the audio element.</audio> |

|Separated Guitar| Separated Piano|
| :----------: | :----------: |
|<audio controls preload="none"><source src="media/separation/2/sep-guitar.mp3" type="audio/mpeg"> Your browser does not support the audio element.</audio> |    <audio controls preload="none"><source src="media/separation/2/sep-piano.mp3" type="audio/mpeg"> Your browser does not support the audio element.</audio>|
