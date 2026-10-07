"""Convolutional models.

Start from an ImageNet-pretrained backbone with a regression head and a sigmoid
output, then improve it. Watch the first epochs: on a target that sits mostly
near 1, a freshly initialised sigmoid head starts at 0.5 and can take a long
time to leave it.

Input resolution is the first thing to experiment with. At 384 px a ResNet-18
loses to the handcrafted baseline; the defects that matter are a few pixels
wide on the 2048 px originals.

Be careful with augmentation. Flips and small translations are safe. Anything
that changes image quality, such as blur, brightness jitter or added noise,
alters the thing you are trying to predict and will corrupt the label.
"""
