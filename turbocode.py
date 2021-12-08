import numpy as np   ###good
import random
import commpy.channelcoding.turbo as dturbo
import commpy.channelcoding.convcode as cc
import commpy.channelcoding.interleavers as RandInterlv
import time

class turbo():
    def __init__(self,args):
        M=np.array([args.M])
        generator_matrix=np.array([[args.enc1,args.enc2]])
        feedback=args.feedback
        self.trellis1=cc.Trellis(M,generator_matrix,feedback=feedback)
        # trellis data structure
        self.trellis2=cc.Trellis(M,generator_matrix,feedback=feedback)
        # trellis data structure
        self.interleaver=RandInterlv.RandInterlv(args.block_len,0)
        self.p_array=self.interleaver.p_array
        self.p_array=self.p_array.astype(np.int32)
        self.args=args
        print('[Convolutional Code Codec] Encoder', 'M ', M, ' Generator Matrix ', generator_matrix, ' Feedback ', feedback)
    def encoder(self, x):
        [sys,par1,par2]=dturbo.turbo_encode(x,self.trellis1,self.trellis2,self.interleaver)
        # code_rate=3
        return (sys,par1,par2)
    def decoder(self,y,snr):
        sys_r,par1_r,par2_r=y[:,0],y[:,1],y[:,2]
        # npower=self.noisepower(y,snr)
        tmp=dturbo.hazzys_g_turbo_decode(sys_r, par1_r, par2_r, self.trellis1,snr, self.args.num_iteration , self.interleaver, L_int = None)
        tmp=tmp.reshape((-1,1))
        return tmp
    def interleaver(self,length):
        random.seed(10)
        tmp=random.sample(range(1,int(length)+1),int(length))
        interleaverindex=np.array(tmp,dtype="float64").reshape((1,-1))
        return interleaverindex
    def noisepower(self,x,snr):
        # sigpower=np.sum(np.abs(x)**2,axis=(0,1))/(x.shape[0]*x.shape[1])
        sigpower=np.sum(x**2)/(x.shape[0]*x.shape[1])
        snr=10**(snr/10)
        npower=sigpower/snr
        return npower
        
if __name__ == "__main__":
    puncture=0.0
    coderate=1/(2+puncture)
    bitlen=398
    args={
        "g":[[1,1,1],[1,0,1]],
        "puncture":puncture,
        "coderate":coderate,
        "dec_alg":"sova",
        "niter":5.0,
        "bitlen":bitlen,
    }
    snrlist=[i for i in range(-10,10,5)]
    turbo=classicalturbo(args)
    for snr in snrlist:
        msgint=np.random.randint(0,2,(bitlen,1))
        msgint=np.array(msgint,dtype="float64")
        encodebits=turbo.encoder(msgint)
        decodebits=turbo.decoder(encodebits,snr)
        ber=np.sum(np.logical_xor(msgint,decodebits))/bitlen
        print("snr=",snr,"ber=",ber)
