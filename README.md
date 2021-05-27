# TalkNet 2 [WIP]
[TalkNet 2: Non-Autoregressive Depth-Wise Separable Convolutional Model for Speech Synthesis with Explicit Pitch and Duration Prediction.](https://arxiv.org/abs/2104.08189)

<br />Official TalkNet 2 repo [here](https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/tts/models/talknet.py)

## Work remains:
- [x] Add masking to all QuartzNet Blocks.
- [ ] Add PostNet to Mel-Spectrogram generator.
- [ ] Clean up and modify all model implementation as per best practices.
- [ ] Add Text and Audio processing code.
- [ ] Add dataloader and training code.
- [ ] Test the whole Talknet2 setup and post result.


## Citation:
```
@misc{beliaev2021talknet,
      title={TalkNet 2: Non-Autoregressive Depth-Wise Separable Convolutional Model Stanislav Beliaev, Boris Ginsburgfor Speech Synthesis with Explicit Pitch and Duration Prediction}, 
      author={Stanislav Beliaev and Boris Ginsburg},
      year={2021},
      eprint={2104.08189},
      archivePrefix={arXiv},
      primaryClass={eess.AS}
}
```
