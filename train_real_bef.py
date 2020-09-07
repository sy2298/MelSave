import os, argparse, traceback, glob, librosa, random, itertools, scipy, time, torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from gan import Generator, MultiScale
from pathlib import Path
import matplotlib.pyplot as plt
import numpy
from scipy.io import wavfile
import torch
from pytorch_wavelets import DWTForward, DWTInverse
import pywt

class MelDataset(Dataset):
    def __init__(self, seq_len, mel_list, hop_length):
        self.seq_len = seq_len
        self.mel_list = mel_list
        self.hop_length = hop_length

    def __len__(self):
        return len(self.mel_list)

    def __getitem__(self, idx):
        #print("get item의 idx값" ,idx)
        mel = torch.load(self.mel_list[idx])
        start = random.randint(0, mel.size(1) - self.seq_len - 1)
        mel = mel[:, start : start + self.seq_len]  # (80, 32)
        #print(idx,"번째 ",start,self.seq_len,"->",start+self.seq_len,"첫번째값:",mel[0])

        wav_name = self.mel_list[idx].replace('.mel', '.wav')
        wav, _ = librosa.core.load(wav_name, sr=None)
        wav = torch.from_numpy(wav).float()
        start *= args.hop_length
        wav = wav[start : start + self.seq_len * args.hop_length]  # (1, 32 * 256)

        return mel, wav.unsqueeze(0)


def train(args):
    mel_list = glob.glob(os.path.join(args.train_dir, '*.mel'))
    trainset = MelDataset(args.seq_len, mel_list, args.hop_length)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, num_workers=0, shuffle=False, drop_last=True)

    test_mel = glob.glob(os.path.join(args.valid_dir, '*.mel'))
    print(args.valid_dir)
    print(type(test_mel))
    print(test_mel[0])
    testset = []
    for i in range(args.test_num):
        mel = torch.load(test_mel[i])
        mel = mel[:, :args.test_len]
        mel = mel.unsqueeze(0)
        testset.append(mel)

    #print(testset[0].shape)
    G = Generator(80)
    D = MultiScale()

    G = G.cuda()
    D = D.cuda()


    g_optimizer = optim.Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.9))
    d_optimizer = optim.Adam(D.parameters(), lr=1e-4, betas=(0.5, 0.9))

    step, epochs = 0, 0
    if args.load_dir is not None:
        print("Loading checkpoint")
        ckpt = torch.load(args.load_dir)
        G.load_state_dict(ckpt['G'])
        g_optimizer.load_state_dict(ckpt['g_optimizer'])
        D.load_state_dict(ckpt['D'])
        d_optimizer.load_state_dict(ckpt['d_optimizer'])
        step = ckpt['step']
        epochs = ckpt['epoch']
        print('Load Status: Epochs %d, Step %d' % (epochs, step))

    torch.backends.cudnn.benchmark = True
    start = time.time()
    try:
        checks=1
        for epoch in itertools.count(1):
            for idx,(mel, audio) in enumerate(train_loader):
                #('audio : ',audio)
                mel = mel.cuda()
                coeffs=pywt.wavedec(audio,'db1',level=2,mode='periodic')
                c1,c2,c3=coeffs

                new_c3 = []

                c3 = c3[0][0]
                c2 = c2[0][0]
                c1 = c1[0][0]
                sum = 0

                #avg pooling ---- c3
                for i in range(len(c3)):
                    if i % 2 == 0:
                        sum += c3[i]
                        new_c3.append(sum / 2)
                        sum = 0
                    else :
                        sum += c3[i]

                #upsample c2
                new_c2 = []
                for i in range(len(c2)):
                    if i == len(c2)-1:
                        new_c2.append(c2[i])
                        new_c2.append(c2[i])
                    else :
                        new_c2.append(c2[i])
                        new_c2.append( (c2[i] + c2[i+1]) / 2 )


                temp=numpy.concatenate((c1,new_c2),axis=0)
                temp=numpy.concatenate((temp,new_c3),axis=0)
                arr2 = numpy.array([[temp]])
                arr3 = torch.from_numpy(arr2).float()



                fake_audio = arr3.cuda()

                audio = arr3.cuda()

                # Discriminator
                d_real = D(audio)
                #print(type(d_real))
                d_loss_real = 0
                for scale in d_real:
                    d_loss_real += F.relu(1 - scale[-1]).mean()

                fake_audio = G(mel)
                #############################
                #Wavelet Transform
                d_fake = D(fake_audio.cuda().detach())
                '''
                f_coeffs=pywt.wavedec(fake_audio,'db1',level=2,mode='periodic')
                f_c1,f_c2,f_c3=f_coeffs

                f_c1 = f_c1[0][0]
                f_c2 = f_c2[0][0]
                f_c3 = f_c3[0][0]
                sum = 0

                new_f_c3 = []
                #avg pooling ---- c3
                for i in range(len(f_c3)):
                    if i % 2 == 0:
                        sum += f_c3[i]
                        new_f_c3.append(sum / 2)
                        sum = 0
                    else :
                        sum += f_c3[i]

                #upsample c2
                new_f_c2 = []
                for i in range(len(f_c2)):
                    if i == len(f_c2)-1:
                        new_f_c2.append(f_c2[i])
                        new_f_c2.append(f_c2[i])
                    else :
                        new_f_c2.append(f_c2[i])
                        new_f_c2.append( (f_c2[i] + f_c2[i+1]) / 2 )


                temp=numpy.concatenate((f_c1,new_f_c2),axis=0)
                temp=numpy.concatenate((temp,new_f_c3),axis=0)

                arr2 = numpy.array([[temp]])
                arr3 = torch.from_numpy(arr2).float()

                fake_audio = arr3.cuda()
                '''

                #print(len(d_fake[0]))  -----> 출력값 : 7
                #print(len(d_fake))     -----> 출력값 : 3
                #d_fake = D(fake_audio)
                d_loss_fake = 0
                for scale in d_fake:
                    d_loss_fake += F.relu(1 + scale[-1]).mean()

                d_loss = d_loss_real + d_loss_fake

                D.zero_grad()
                d_loss.backward()
                d_optimizer.step()

                # Generator
                d_fake = D(fake_audio.cuda())
                g_loss = 0
                for scale in d_fake:
                    g_loss += -scale[-1].mean()
                #print(g_loss)
                # Feature Matching
                feature_loss = 0
                # feat_weights = 4.0 / 5.0  # discriminator block size + 1
                # D_weights = 1.0 / 3.0  # multi scale size
                # wt = D_weights * feat_weights  # not in paper
                for i in range(1):
                    for j in range(len(d_fake[i]) - 1):
                        feature_loss += F.l1_loss(d_fake[i][j], d_real[i][j].detach())

                g_loss += args.lambda_feat * feature_loss

                G.zero_grad()
                g_loss.backward()
                g_optimizer.step()

                step += 1
                if step % args.log_interval != 0:
                    print('Epoch: %-5d, Step: %-7d, D_loss: %.05f, G_loss: %.05f, ms/batch: %5.2f' %
                        (epoch, step, d_loss, g_loss, 1000 * (time.time() - start) / args.log_interval))
                    start = time.time()
                '''
                if step % args.save_interval == 0:
                    root = Path(args.save_dir)
                    with torch.no_grad():
                        for i, mel_test in enumerate(testset):
                            g_audio = G(mel_test.cuda())
                            g_audio = g_audio.squeeze().cpu()
                            audio = (g_audio.numpy() * 32768)
                            scipy.io.wavfile.write(root / ('generated-%d-%dk-%d.wav' % (epoch, step // 1000, i)),
                                                   22050,
                                                   audio.astype('int16'))
                '''
                if step % 1000 == 0:
                    #print('ddddd')
                    root = Path(args.save_dir)
                    with torch.no_grad():

                        for i, mel_test in enumerate(testset):
                            g_audio = G(mel_test.cuda())
                            g_audio = g_audio.squeeze().cpu().clone().numpy()

                            std_=int(g_audio.shape[0]/4)
                            g1 = g_audio[0:std_]
                            g2 = g_audio[std_:std_*3]
                            # 반으로 줄이고
                            new_g2 = []
                            for k in range(len(g2)):
                                if k % 2 == 0:
                                    sum += g2[k]
                                    new_g2.append(sum / 2)
                                    sum = 0
                                else :
                                    sum += g2[k]


                            g3 = g_audio[std_*3:std_*4]
                            # 두배로 늘리고
                            #upsample c2
                            new_g3 = []
                            for k in range(len(g3)):
                                if k == len(g3)-1:
                                    new_g3.append(g3[k])
                                    new_g3.append(g3[k])
                                else :
                                    new_g3.append(g3[k])
                                    new_g3.append( (g3[k] + g3[k+1]) / 2 )


                            new_g2 = numpy.array(new_g2)
                            new_g3 = numpy.array(new_g3)

                            coeffs_=[ g1, new_g2, new_g3 ]

                            y = pywt.waverec(coeffs_,'db1',mode='periodic')
                           # y = numpy.asarray(y,dtype=numpy.int16)
                            y = numpy.int16(y/numpy.max(numpy.abs(y))*32767)

                            wavfile.write(root / ('wavelet-%d-%dk-%d.wav' % (epoch, step //1000 , i)),
                                                   22050, y)

                        for i, mel_test in enumerate(testset):
                            g_audio = G(mel_test.cuda())
                            g_audio = g_audio.squeeze().cpu()
                            audio = (g_audio.numpy() * 32768)

                            scipy.io.wavfile.write(root / ('no_wavelet-%d-%dk-%d.wav' % (epoch, step//1000, i)),
                                                   22050,
                                                   audio.astype('int16'))

                    print("Saving checkpoint")
                    torch.save({
                        'G': G.state_dict(),
                        'g_optimizer': g_optimizer.state_dict(),
                        'D': D.state_dict(),
                        'd_optimizer': d_optimizer.state_dict(),
                        'step': step,
                        'epoch': epoch,
                    }, root / ('ckpt-%dk.pt' % (step // 1000)))



    except Exception as e:
        traceback.print_exc()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", default='./logs')
    parser.add_argument("--load_dir", default=None)
    parser.add_argument("--train_dir", default='./train')
    parser.add_argument("--valid_dir", default='./valid')
    parser.add_argument("--test_num", default=2)
    parser.add_argument("--hop_length", default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--test_len", type=int, default=300)
    parser.add_argument("--lambda_feat", default=10)
    parser.add_argument("--save_interval", type=int, default=50000)
    parser.add_argument("--log_interval", type=int, default=1000)
    args = parser.parse_args()
    save_dir = os.path.join(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)
    train(args)
