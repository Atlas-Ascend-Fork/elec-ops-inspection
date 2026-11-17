#pragma once
#include "kernel_operator.h"
#include "fdksa_plan.h"
using namespace AscendC;
namespace fdksa {
constexpr uint32_t kMaxSpatial=72*96;
constexpr uint32_t kHalo=kMaxSpatial+2*96;
template<HardEvent E> __aicore__ inline void Fence(){SetFlag<E>(EVENT_ID0);WaitFlag<E>(EVENT_ID0);}
__aicore__ inline void WriteResult(GlobalTensor<float> kg,GlobalTensor<float> sg,
 LocalTensor<float> out,uint32_t channel,float x,float y,float score){
 out.SetValue(0,x);out.SetValue(1,y);out.SetValue(8,score);
 Fence<HardEvent::S_MTE3>();
 DataCopyPad(kg[channel*2],out,DataCopyExtParams{1,8,0,0,0});
 DataCopyPad(sg[channel],out[8],DataCopyExtParams{1,4,0,0,0});
}
__aicore__ inline bool ZeroWeights(LocalTensor<float> w,LocalTensor<float> temp,LocalTensor<float> out){
 ShiftLeft(temp.ReinterpretCast<uint32_t>(),w.ReinterpretCast<uint32_t>(),uint32_t(1),288);
 PipeBarrier<PIPE_V>();
 Cast(temp[320],temp.ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,288);
 PipeBarrier<PIPE_V>();Abs(temp[320],temp[320],288);PipeBarrier<PIPE_V>();
 ReduceMax(out,temp[320],temp[640],288,false);Fence<HardEvent::V_S>();
 return out.GetValue(0)==0.0f;
}
__aicore__ inline bool FiniteFeatures(GlobalTensor<float> fg,LocalTensor<float> x,
 LocalTensor<float> scratch,LocalTensor<float> out,uint32_t start,uint32_t count){
 constexpr uint32_t chunk=32768;
 for(uint32_t offset=0;offset<count;offset+=chunk){
  uint32_t size=count-offset<chunk?count-offset:chunk;
  DataCopy(x,fg[start+offset],size);Fence<HardEvent::MTE2_V>();
  Abs(x,x,size);PipeBarrier<PIPE_V>();
  auto flags=scratch.ReinterpretCast<uint8_t>();
  CompareScalar(flags,x,3.402823466e38f,CMPMODE::LE,size);PipeBarrier<PIPE_V>();
  auto words=flags.ReinterpretCast<uint16_t>();
  Not(words,words,size/16);PipeBarrier<PIPE_V>();
  Cast(scratch[2048],words.ReinterpretCast<int16_t>(),RoundMode::CAST_NONE,size/16);PipeBarrier<PIPE_V>();
  Abs(scratch[2048],scratch[2048],size/16);PipeBarrier<PIPE_V>();
  ReduceMax(out,scratch[2048],scratch[4096],size/16,false);Fence<HardEvent::V_S>();
  float maximum=out.GetValue(0);
  if(maximum!=0.0f)return false;
  Fence<HardEvent::V_MTE2>();
 }
 return true;
}
// Zero weights make all four heatmaps constant. One owner per batch checks
// that batch's features and writes all four outputs; no shared state is needed.
__aicore__ inline int UniformBatch(GM_ADDR feat,GM_ADDR weight,GM_ADDR bias,
 GM_ADDR keypoints,GM_ADDR scores,const FdksaPlan& p,uint32_t channel,TPipe& pipe){
 TBuf<TPosition::VECCALC> storage;pipe.InitBuffer(storage,48128*4);
 auto base=storage.Get<float>();auto scratch=base[32768];auto w=base[45056];
 auto b=base[46336];auto out=base[46400];
 GlobalTensor<float> fg,wg,bg,kg,sg;
 fg.SetGlobalBuffer((__gm__ float*)feat);wg.SetGlobalBuffer((__gm__ float*)weight);bg.SetGlobalBuffer((__gm__ float*)bias);
 kg.SetGlobalBuffer((__gm__ float*)keypoints);sg.SetGlobalBuffer((__gm__ float*)scores);
 DataCopy(w,wg,1152);DataCopyPad(b,bg,DataCopyExtParams{1,16,0,0,0},DataCopyPadExtParams<float>{true,0,4,0.0f});
 Fence<HardEvent::MTE2_V>();
 ShiftLeft(scratch.ReinterpretCast<uint32_t>(),w.ReinterpretCast<uint32_t>(),uint32_t(1),1152);PipeBarrier<PIPE_V>();
 Cast(scratch[1152],scratch.ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,1152);PipeBarrier<PIPE_V>();
 Abs(scratch[1152],scratch[1152],1152);PipeBarrier<PIPE_V>();
 ReduceMax(out,scratch[1152],scratch[2304],1152,false);Fence<HardEvent::V_S>();
 if(out.GetValue(0)!=0.0f)return 0;
 Fence<HardEvent::MTE2_S>();
 for(uint32_t c=0;c<4;++c){float value=b.GetValue(c);if(!(value<=3.402823466e38f && value>=-3.402823466e38f))return 0;}
 if(channel%4!=0)return 1;
 const uint32_t n=p.height*p.width;
 if(!FiniteFeatures(fg,base,scratch,out,(channel/4)*32*n,32*n))return 2;
 const float x=float(int32_t(p.width))*0.5f*float(int32_t(p.stride));
 const float y=float(int32_t(p.height))*0.5f*float(int32_t(p.stride));
 for(uint32_t c=0;c<4;++c){out.SetValue(c*2,x);out.SetValue(c*2+1,y);out.SetValue(8+c,b.GetValue(c));}
 Fence<HardEvent::S_MTE3>();DataCopy(kg[channel*2],out,8);DataCopyPad(sg[channel],out[8],DataCopyExtParams{1,16,0,0,0});
 return 1;
}
constexpr uint32_t kConvTile=4096,kConvHalo=kConvTile+2*96+16;
__aicore__ inline void LoadChannel(LocalTensor<float> halo,GlobalTensor<float> fg,
 uint32_t start,uint32_t count,uint32_t top,uint32_t width,event_t ev){
 DataCopyPad(halo[top*width],fg[start],DataCopyExtParams{1,(count-1)*4,0,0,0},DataCopyPadExtParams<float>{true,1,0,0.0f});
 DataCopy(halo[kConvHalo+top*width],fg[start],count);
 DataCopyPad(halo[2*kConvHalo+top*width],fg[start+1],DataCopyExtParams{1,(count-1)*4,0,0,0},DataCopyPadExtParams<float>{true,0,1,0.0f});
 SetFlag<HardEvent::MTE2_V>(ev);
}
__aicore__ inline void Run(GM_ADDR feat,GM_ADDR weight,GM_ADDR bias,GM_ADDR keypoints,
 GM_ADDR scores,const FdksaPlan& p,TPipe& pipe,uint32_t channel){
 TBuf<TPosition::VECCALC> storage;pipe.InitBuffer(storage,48128*4);
 auto base=storage.Get<float>();auto h=base;auto halo=base[6912];auto t=base[32768];
 auto w=base[47104];auto b=base[47424];auto o=base[47432];
 GlobalTensor<float> fg,wg,bg,kg,sg;
 fg.SetGlobalBuffer((__gm__ float*)feat);wg.SetGlobalBuffer((__gm__ float*)weight);
 bg.SetGlobalBuffer((__gm__ float*)bias);kg.SetGlobalBuffer((__gm__ float*)keypoints);sg.SetGlobalBuffer((__gm__ float*)scores);
 const uint32_t n=p.height*p.width,batch=channel/4,oc=channel%4;
 DataCopy(w,wg[oc*288],288);
 DataCopyPad(b,bg,DataCopyExtParams{1,16,0,0,0},DataCopyPadExtParams<float>{true,0,4,0.0f});
 Fence<HardEvent::MTE2_V>();
 bool zero=ZeroWeights(w,t,o);
 Fence<HardEvent::MTE2_S>();const float biasValue=b.GetValue(oc);
 if(zero && biasValue<=3.402823466e38f && biasValue>=-3.402823466e38f &&
    FiniteFeatures(fg,base,base[32768],o,batch*32*n,32*n)){
  WriteResult(kg,sg,o,channel,float(int32_t(p.width))*0.5f*float(int32_t(p.stride)),
   float(int32_t(p.height))*0.5f*float(int32_t(p.stride)),biasValue);
  return;
 }
 Duplicate(h,0.0f,n);
 const uint32_t rowsPerTile=kConvTile/p.width;
 auto leftBits=base[32768].ReinterpretCast<uint16_t>();
 auto rightBits=base[32768+kConvHalo].ReinterpretCast<uint16_t>();
 const uint32_t maskCount=(rowsPerTile+2)*p.width;
 Duplicate(leftBits,uint16_t(65535),maskCount*2);
 Duplicate(rightBits,uint16_t(65535),maskCount*2);PipeBarrier<PIPE_V>();
 uint64_t leftEdge[2]={1,0},rightEdge[2]={128,0};
 Duplicate(leftBits.ReinterpretCast<float>(),0.0f,leftEdge,uint8_t(rowsPerTile+2),uint16_t(1),uint8_t(p.width/8));
 Duplicate(rightBits.ReinterpretCast<float>()[p.width-8],0.0f,rightEdge,uint8_t(rowsPerTile+2),uint16_t(1),uint8_t(p.width/8));
 PipeBarrier<PIPE_V>();
 for(uint32_t y0=0;y0<p.height;y0+=rowsPerTile){
  const uint32_t outputRows=p.height-y0<rowsPerTile?p.height-y0:rowsPerTile;
  const uint32_t count=outputRows*p.width,top=y0==0?1:0,bottom=y0+outputRows==p.height?1:0;
  const uint32_t inputCount=(outputRows+2-top-bottom)*p.width,sy=y0==0?0:y0-1;
  Duplicate(halo,0.0f,6*kConvHalo);Fence<HardEvent::V_MTE2>();
  LoadChannel(halo,fg,batch*32*n+sy*p.width,inputCount,top,p.width,EVENT_ID0);
  for(uint32_t c=0;c<32;++c){
   const uint32_t slot=c%2,next=1-slot;
   if(c+1<32){
    if(c>0)WaitFlag<HardEvent::V_MTE2>(static_cast<event_t>(next));
    LoadChannel(halo[next*3*kConvHalo],fg,(batch*32+c+1)*n+sy*p.width,inputCount,top,p.width,static_cast<event_t>(next));
   }
   WaitFlag<HardEvent::MTE2_V>(static_cast<event_t>(slot));
   auto current=halo[slot*3*kConvHalo];
   const uint32_t validBits=(outputRows+2)*p.width*2;
   And(current.ReinterpretCast<uint16_t>(),current.ReinterpretCast<uint16_t>(),leftBits,validBits);
   And(current[2*kConvHalo].ReinterpretCast<uint16_t>(),current[2*kConvHalo].ReinterpretCast<uint16_t>(),rightBits,validBits);
   PipeBarrier<PIPE_V>();
   #pragma unroll
   for(uint32_t ky=0;ky<3;++ky){
    #pragma unroll
    for(uint32_t kx=0;kx<3;++kx){
     Axpy(h[y0*p.width],current[kx*kConvHalo+ky*p.width],w.GetValue(c*9+ky*3+kx),count);
     PipeBarrier<PIPE_V>();
    }
   }
   SetFlag<HardEvent::V_MTE2>(static_cast<event_t>(slot));
  }
  WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
 }
 Adds(h,h,biasValue,n);PipeBarrier<PIPE_V>();
 ReduceMax(o,h,t,n,false);Fence<HardEvent::V_S>();const float score=o.GetValue(0);
 auto u=halo;auto v=halo[kHalo];auto scratch=halo[2*kHalo];
 CreateVecIndex(u,0.0f,p.width);CreateVecIndex(v,0.0f,n);PipeBarrier<PIPE_V>();
 for(uint32_t start=0;start<p.width;start+=64){
  uint64_t mask=p.width-start<64?p.width-start:64;
  Adds(u[p.width+start],u[start],0.0f,mask,uint8_t(p.height-1),UnaryRepeatParams{1,1,uint8_t(p.width/8),0});
 }
 Duplicate(t,p.tau,n);PipeBarrier<PIPE_V>();
 Div(o,o,t,1);Fence<HardEvent::V_S>();const float maximum=o.GetValue(0);
 if(maximum>3.402823466e38f || maximum<-3.402823466e38f){
  Adds(h,h,-score,n);PipeBarrier<PIPE_V>();Div(h,h,t,n);
 }else{
  Div(h,h,t,n);PipeBarrier<PIPE_V>();Adds(h,h,-maximum,n);
 }
 PipeBarrier<PIPE_V>();Exp(h,h,n);PipeBarrier<PIPE_V>();
 ReduceSum(o,h,scratch,n);Fence<HardEvent::V_S>();const float z=o.GetValue(0);
 Mul(t,h,u,n);PipeBarrier<PIPE_V>();ReduceSum(o,t,scratch,n);Fence<HardEvent::V_S>();const float sx=o.GetValue(0);
 Mul(t,h,v,n);PipeBarrier<PIPE_V>();ReduceSum(o,t,scratch,n);Fence<HardEvent::V_S>();const float si=o.GetValue(0);
 WriteResult(kg,sg,o,channel,(sx/z+0.5f)*float(int32_t(p.stride)),
  ((si-sx)/(z*float(int32_t(p.width)))+0.5f)*float(int32_t(p.stride)),score);
}
}
template<class FdksaMeta>
__aicore__ inline void FdksaCompute(GM_ADDR feat,GM_ADDR weight,GM_ADDR bias,
 GM_ADDR keypoints,GM_ADDR scores,GM_ADDR workspace,const FdksaMeta& data){
 InitSocState();
 const FdksaPlan p{data.batch,data.height,data.width,data.stride,data.tau,uint32_t(GetBlockNum())};
 for(uint32_t channel=GetBlockIdx();channel<p.batch*4;channel+=GetBlockNum()){
  int mode;
  {TPipe pipe;mode=fdksa::UniformBatch(feat,weight,bias,keypoints,scores,p,channel,pipe);}
  if(mode==1)continue;
  const uint32_t last=mode==2?channel+4:channel+1;
  for(uint32_t output=channel;output<last;++output){TPipe pipe;fdksa::Run(feat,weight,bias,keypoints,scores,p,pipe,output);}
 }
}
