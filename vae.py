

def latent2im(decoder,compressed):
    x_decoded = decoder.predict(compressed)
    im = x_decoded[0].reshape(64, 64, 3)
    return im


def im2latent(encoder, im):
    compressed = encoder.predict(im,batch_size=1)
    return compressed