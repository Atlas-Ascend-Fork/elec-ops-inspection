/*
 * Copyright (c) 2026 Delicate02. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include "kernel_operator.h"

// Compile-time UB address descriptors only; no input values or output values.
namespace fire_metadata {
template<unsigned Words> struct AddressPacket { uint64_t words[Words]{}; };
struct AddressContext { uint64_t source[16]{},destination[16]{}; };

template<unsigned Words>
constexpr void PutContext(AddressPacket<Words>& packet,unsigned index,const AddressContext& context) {
    for(unsigned side=0;side<2;++side) {
        const auto* addresses=side?context.destination:context.source;
        for(unsigned part=0;part<2;++part) for(unsigned word=0;word<2;++word) {
            uint64_t packed=0;
            for(unsigned lane=0;lane<4;++lane)
                packed|=((addresses[part*8+word*4+lane]>>5)&0x1fffULL)<<(16*lane);
            packet.words[index*16+side*8+part*4+word]=packed;
        }
    }
}

constexpr AddressContext RawHalf(unsigned pitch,unsigned width,uint64_t source,uint64_t destination,unsigned block) {
    AddressContext context{};
    for(unsigned i=0;i<16;++i) {
        const unsigned j=block*16+i;
        context.source[i]=source+2*(i*48+block*16);
        context.destination[i]=destination+2*((j%6)*pitch+(j/6)*width);
    }
    return context;
}

constexpr AddressContext RawFloat(unsigned pitch,unsigned width,uint64_t source,uint64_t destination,unsigned block) {
    AddressContext context{};
    for(unsigned i=0;i<16;++i)context.source[i]=source+4*(i*48+block*8);
    for(unsigned i=0;i<8;++i) {
        const unsigned j=block*8+i,offset=(j%6)*pitch+(j/6)*width;
        context.destination[2*i]=destination+4*offset;
        context.destination[2*i+1]=destination+4*(offset+8);
    }
    return context;
}

constexpr AddressContext Prior(unsigned pitch,unsigned width,uint64_t source,unsigned block) {
    AddressContext context{};
    for(unsigned i=0;i<16;++i)context.source[i]=source+4*(i*40+block*8);
    for(unsigned i=0;i<8;++i) {
        const unsigned j=block*8+i,offset=(j%5)*pitch+(j/5)*width;
        context.destination[2*i]=4*offset;context.destination[2*i+1]=4*(offset+8);
    }
    return context;
}

constexpr AddressContext Output(unsigned pitch,unsigned width,uint64_t source,uint64_t destination,unsigned block) {
    AddressContext context{};
    for(unsigned i=0;i<8;++i) {
        const unsigned j=block*8+i,offset=(j%5)*pitch+(j/5)*width;
        context.source[i]=source+4*offset;context.source[i+8]=source+4*(offset+8);
        context.destination[2*i]=destination+4*(i*40+block*8);
        context.destination[2*i+1]=destination+4*((i+8)*40+block*8);
    }
    return context;
}

template<unsigned Tile,bool NativeHalf,bool Large=false>
constexpr AddressPacket<Large?336:256> MakeContexts() {
    constexpr unsigned Q=(Tile+127)/128*16,P=8*Q+(NativeHalf?16:24);
    constexpr uint64_t S0=5*P*4+128,V0=S0+6*P*4+128,O0=V0+6*P*4+128,F0=O0+6*P*4+128;
    AddressPacket<Large?336:256> packet{};
    if constexpr(NativeHalf) {
        for(unsigned block=0;block<3;++block)PutContext(packet,block,RawHalf(P,Q,V0,O0,block));
    } else {
        for(unsigned block=0;block<6;++block)PutContext(packet,block,RawFloat(P,Q,O0,S0,block));
    }
    for(unsigned block=0;block<5;++block) {
        PutContext(packet,6+block,Prior(P,Q,F0,block));
        PutContext(packet,11+block,Output(P,Q,S0,F0,block));
        if constexpr(Large)PutContext(packet,16+block,Output(P,Q,S0,V0,block));
    }
    return packet;
}

template<unsigned Tile>
constexpr AddressPacket<288> MakeLargeHalf() {
    constexpr unsigned Q=(Tile+127)/128*16,P=8*Q+16;
    constexpr uint64_t S0=5*P*4+448,V0=S0+6*P*4+128,O0=V0+6*P*4+128,F0=O0+6*P*4+128;
    AddressPacket<288> packet{};
    for(unsigned block=0;block<3;++block)PutContext(packet,block,RawHalf(P,Q,O0,V0,block));
    for(unsigned block=0;block<5;++block) {
        PutContext(packet,3+block,Prior(P,Q,F0,block));
        PutContext(packet,8+block,Output(P,Q,S0,F0,block));
        PutContext(packet,13+block,Output(P,Q,S0,V0,block));
    }
    return packet;
}

template<unsigned Capacity>
constexpr AddressPacket<Capacity> MakeSmall() {
    AddressPacket<Capacity> packet{};
    for(unsigned i=0;i<Capacity/2;++i) {
        packet.words[i]=uint64_t(2*i*24)|(uint64_t((2*i+1)*24)<<32);
        packet.words[Capacity/2+i]=uint64_t(2*i*20)|(uint64_t((2*i+1)*20)<<32);
    }
    return packet;
}

template<unsigned Kind,unsigned Tile> struct Table;
static const __gm__ AddressPacket<32> fire_address_0_32 __attribute__((aligned(512))) = MakeSmall<32>();
template<> struct Table<0,32> {
    __aicore__ static inline __gm__ uint64_t* Address() { return const_cast<__gm__ uint64_t*>(fire_address_0_32.words); }
};
static const __gm__ AddressPacket<64> fire_address_0_64 __attribute__((aligned(512))) = MakeSmall<64>();
template<> struct Table<0,64> {
    __aicore__ static inline __gm__ uint64_t* Address() { return const_cast<__gm__ uint64_t*>(fire_address_0_64.words); }
};
static const __gm__ AddressPacket<128> fire_address_0_128 __attribute__((aligned(512))) = MakeSmall<128>();
template<> struct Table<0,128> {
    __aicore__ static inline __gm__ uint64_t* Address() { return const_cast<__gm__ uint64_t*>(fire_address_0_128.words); }
};
static const __gm__ AddressPacket<256> fire_address_1_512 __attribute__((aligned(512))) = MakeContexts<512,false>();
template<> struct Table<1,512> {
    __aicore__ static inline __gm__ uint64_t* Address() { return const_cast<__gm__ uint64_t*>(fire_address_1_512.words); }
};
static const __gm__ AddressPacket<256> fire_address_2_512 __attribute__((aligned(512))) = MakeContexts<512,true>();
template<> struct Table<2,512> {
    __aicore__ static inline __gm__ uint64_t* Address() { return const_cast<__gm__ uint64_t*>(fire_address_2_512.words); }
};
static const __gm__ AddressPacket<256> fire_address_1_528 __attribute__((aligned(512))) = MakeContexts<528,false>();
template<> struct Table<1,528> {
    __aicore__ static inline __gm__ uint64_t* Address() { return const_cast<__gm__ uint64_t*>(fire_address_1_528.words); }
};
static const __gm__ AddressPacket<256> fire_address_2_528 __attribute__((aligned(512))) = MakeContexts<528,true>();
template<> struct Table<2,528> {
    __aicore__ static inline __gm__ uint64_t* Address() { return const_cast<__gm__ uint64_t*>(fire_address_2_528.words); }
};
static const __gm__ AddressPacket<256> fire_address_1_640 __attribute__((aligned(512))) = MakeContexts<640,false>();
template<> struct Table<1,640> {
    __aicore__ static inline __gm__ uint64_t* Address() { return const_cast<__gm__ uint64_t*>(fire_address_1_640.words); }
};
static const __gm__ AddressPacket<256> fire_address_2_640 __attribute__((aligned(512))) = MakeContexts<640,true>();
template<> struct Table<2,640> {
    __aicore__ static inline __gm__ uint64_t* Address() { return const_cast<__gm__ uint64_t*>(fire_address_2_640.words); }
};
static const __gm__ AddressPacket<256> fire_address_1_896 __attribute__((aligned(512))) = MakeContexts<896,false>();
template<> struct Table<1,896> {
    __aicore__ static inline __gm__ uint64_t* Address() { return const_cast<__gm__ uint64_t*>(fire_address_1_896.words); }
};
static const __gm__ AddressPacket<256> fire_address_2_896 __attribute__((aligned(512))) = MakeContexts<896,true>();
template<> struct Table<2,896> {
    __aicore__ static inline __gm__ uint64_t* Address() { return const_cast<__gm__ uint64_t*>(fire_address_2_896.words); }
};
static const __gm__ AddressPacket<256> fire_address_1_1024 __attribute__((aligned(512))) = MakeContexts<1024,false>();
template<> struct Table<1,1024> {
    __aicore__ static inline __gm__ uint64_t* Address() { return const_cast<__gm__ uint64_t*>(fire_address_1_1024.words); }
};
static const __gm__ AddressPacket<256> fire_address_2_1024 __attribute__((aligned(512))) = MakeContexts<1024,true>();
template<> struct Table<2,1024> {
    __aicore__ static inline __gm__ uint64_t* Address() { return const_cast<__gm__ uint64_t*>(fire_address_2_1024.words); }
};
static const __gm__ AddressPacket<336> fire_address_3_1640 __attribute__((aligned(512))) = MakeContexts<1640,false,true>();
template<> struct Table<3,1640> {
    __aicore__ static inline __gm__ uint64_t* Address() { return const_cast<__gm__ uint64_t*>(fire_address_3_1640.words); }
};
static const __gm__ AddressPacket<288> fire_address_4_1640 __attribute__((aligned(512))) = MakeLargeHalf<1640>();
template<> struct Table<4,1640> {
    __aicore__ static inline __gm__ uint64_t* Address() { return const_cast<__gm__ uint64_t*>(fire_address_4_1640.words); }
};
static const __gm__ AddressPacket<336> fire_address_3_1664 __attribute__((aligned(512))) = MakeContexts<1664,false,true>();
template<> struct Table<3,1664> {
    __aicore__ static inline __gm__ uint64_t* Address() { return const_cast<__gm__ uint64_t*>(fire_address_3_1664.words); }
};
static const __gm__ AddressPacket<288> fire_address_4_1664 __attribute__((aligned(512))) = MakeLargeHalf<1664>();
template<> struct Table<4,1664> {
    __aicore__ static inline __gm__ uint64_t* Address() { return const_cast<__gm__ uint64_t*>(fire_address_4_1664.words); }
};
} // namespace fire_metadata
