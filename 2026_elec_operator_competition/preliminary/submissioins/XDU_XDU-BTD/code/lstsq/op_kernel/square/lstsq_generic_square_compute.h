// Copyright 2026 XDU_XDU-BTD Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

template<Zq2 StaticN,Zq2 StaticK>
__aicore__ inline void KernelLarge<StaticN, StaticK>::SolveInPlaceWithFactors(Zq3 base,__gm__ float*rhs,Zq2 width)
{
if(UseColumnLocal()){ColumnTriangularAll(base,rhs,rhs,width);return;}
if(UseVectorSolve()&&width<=4u){
VectorTriangularBatch(base,rhs,width,rhs,width,width);
return;
}
if(UseVectorSolve()){
for(Zq2 col=0;col<width;++col)
VectorTriangularColumn(base,rhs+col,width,rhs+col,width);
return;
}
const Zq2 n=N();
TriangularSolveOperations::SolveScalarSquare(*this, base, w_+OffLU(base), w_+OffP(base), rhs, n, width);
}

template<Zq2 StaticN,Zq2 StaticK>
__aicore__ inline void KernelLarge<StaticN, StaticK>::Qr(Zq3 base)
{
    RectangularSolveOperations::Qr(w_+OffLU(base), w_+OffB(base), M(), N(), K());
}
