import numpy as np
import keras
import scipy.io.wavfile
import scipy.signal
import os
import matplotlib.pyplot as plt
import random

# Importing all sound files. Dataset used is TIMIT
cwd = os.getcwd()+'/TIMIT'
soundClips = []
from pathlib import Path
print (cwd)
from sphfile import SPHFile
for soundFile in Path(cwd).glob('**/*.wav'):
    soundClips += [scipy.io.wavfile.read(soundFile)]
dataRate = soundClips[0][0]
soundClips = [i[1] for i in soundClips]


mergedSpeech = np.concatenate(soundClips, axis=0)
validationSpeech = mergedSpeech[mergedSpeech.shape[0] * 9 // 10:]
mergedSpeech = mergedSpeech[:mergedSpeech.shape[0] * 9 // 10]

normalizingFactor=np.std(mergedSpeech)
noisingFactor=.15

clipLength = 1024

denoiserLayers = [
    keras.layers.Input(shape=(clipLength,)),
    keras.layers.Reshape((clipLength, 1)),

    keras.layers.Conv1D(filters=64, kernel_size=25, padding='same'),
    keras.layers.LeakyReLU(),
    keras.layers.Conv1D(filters=64, kernel_size=25, padding='same'),
    keras.layers.LeakyReLU(),
    keras.layers.Dropout(.4),

    keras.layers.Conv1D(filters=64, kernel_size=25, padding='same'),
    keras.layers.LeakyReLU(),
    keras.layers.Conv1D(filters=64, kernel_size=25, padding='same'),
    keras.layers.LeakyReLU(),
    keras.layers.Dropout(.4),

    keras.layers.Conv1D(filters=64, kernel_size=25, padding='same'),
    keras.layers.LeakyReLU(),
    keras.layers.Conv1D(filters=64, kernel_size=25, padding='same'),
    keras.layers.LeakyReLU(),
    keras.layers.Dropout(.4),

    keras.layers.Conv1D(filters=64, kernel_size=25, padding='same'),
    keras.layers.LeakyReLU(),

    keras.layers.Conv1D(filters=1, kernel_size=25, padding='same'),
    keras.layers.Dropout(.4),

    keras.layers.Reshape((clipLength,)),
]


def stackLayers(layerSet):
    stack = layerSet[0]
    for i in range(1, len(layerSet)):
        stack = layerSet[i](stack)
    return stack


def sampleGenerator(originalSound, sampleLength, noisingFactor, batchSize=32, firstFixed=None):
    while 1:
        indices = np.random.randint(low=0, high=originalSound.shape[0] - sampleLength, size=batchSize).tolist()
        if firstFixed != None:
            indices[0] = firstFixed
        samples = np.array([originalSound[index:index + sampleLength] for index in indices])
        noise = np.random.normal(loc=0, scale=2*random.random()*noisingFactor*normalizingFactor, size=samples.shape)
        #print (samples.shape)
        yield ((samples+noise)/normalizingFactor,samples/normalizingFactor)


clipGenerator = sampleGenerator(mergedSpeech, clipLength, noisingFactor, 32)
validationClipGenerator = sampleGenerator(validationSpeech, clipLength, noisingFactor, 256)

denoiser = keras.models.Model(inputs=denoiserLayers[0], outputs=stackLayers(denoiserLayers))
denoiser.compile(optimizer=keras.optimizers.Adam(.0001, decay=1e-7), loss='mse', metrics=['mse', ])
denoiser.summary()

# print(denoiser.evaluate_generator(validationClipGenerator,steps=16))


denoiser.fit_generator(clipGenerator, steps_per_epoch=512, epochs=10, validation_data=validationClipGenerator,
                       validation_steps=16, use_multiprocessing=True,
                       verbose=2)  # Validation batches large for efficiency


def sequentialPredict(data, subsequenceLength, stride):  # Predicts on segments of the data with a (possibly overlapping stride). Averages per-point and returns.
    assert stride <= subsequenceLength
    batchSize = 32
    batchedData = np.empty((int(np.ceil(data.shape[0] - subsequenceLength) / stride) + 1, subsequenceLength))
    startIndices=np.empty((batchedData.shape[0],),dtype=np.int32)
    for i in range(0, batchedData.shape[0]):
        startIndex = i*stride
        if startIndex > data.shape[0] - subsequenceLength:
            startIndex = data.shape[0] - subsequenceLength
        startIndex=int(startIndex)
        batchedData[i] = data[startIndex:startIndex + subsequenceLength] / normalizingFactor
        startIndices[i]=startIndex
    processedBatches = denoiser.predict(batchedData,batch_size=batchSize)*normalizingFactor

    finalData = np.zeros(data.shape)
    hitCounter = np.zeros(data.shape)
    for i in range (0,batchedData.shape[0]):
        finalData[startIndices[i]:startIndices[i]+subsequenceLength]+=processedBatches[i]
        hitCounter[startIndices[i]:startIndices[i]+subsequenceLength]+=1
    finalData=np.divide(finalData,hitCounter)
    return finalData.astype(np.int16)



clip = soundClips[0]
noisedClip = (clip + np.random.normal(loc=0, scale=noisingFactor*normalizingFactor, size=clip.shape)).astype(np.int16)
# print (normalizedClip.shape)


predicted = sequentialPredict(noisedClip, clipLength, stride=clipLength//2)

# print (np.max(predicted),np.std(predicted),np.mean(predicted))
scipy.io.wavfile.write("predicted.wav", dataRate, predicted)
scipy.io.wavfile.write("original.wav", dataRate, soundClips[0])
scipy.io.wavfile.write("noised.wav", dataRate, noisedClip)

width = 1000
baseIndex = 20000

# plt.subplot(411)
# plt.plot(np.arange(width),soundClips[0][baseIndex:baseIndex+width],'b-',np.arange(width),predicted[baseIndex:baseIndex+width],'r-')

plt.subplot(311)
plt.specgram(soundClips[0], Fs=dataRate)
plt.xlabel("Original")

plt.subplot(312)
plt.specgram(noisedClip, Fs=dataRate)
plt.xlabel("Noised")

plt.subplot(313)
plt.specgram(predicted, Fs=dataRate)
plt.xlabel("Predicted")

plt.show()

