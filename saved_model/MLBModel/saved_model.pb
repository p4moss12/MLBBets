Ú¾	
ðÔ
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
Á
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.8.02unknown8®

1st_Hidden_Layer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *(
shared_name1st_Hidden_Layer/kernel

+1st_Hidden_Layer/kernel/Read/ReadVariableOpReadVariableOp1st_Hidden_Layer/kernel*
_output_shapes

: *
dtype0

1st_Hidden_Layer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_name1st_Hidden_Layer/bias
{
)1st_Hidden_Layer/bias/Read/ReadVariableOpReadVariableOp1st_Hidden_Layer/bias*
_output_shapes
: *
dtype0

2nd_Hidden_Layer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *(
shared_name2nd_Hidden_Layer/kernel

+2nd_Hidden_Layer/kernel/Read/ReadVariableOpReadVariableOp2nd_Hidden_Layer/kernel*
_output_shapes
:	 *
dtype0

2nd_Hidden_Layer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_name2nd_Hidden_Layer/bias
|
)2nd_Hidden_Layer/bias/Read/ReadVariableOpReadVariableOp2nd_Hidden_Layer/bias*
_output_shapes	
:*
dtype0

3rd_Hidden_Layer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*(
shared_name3rd_Hidden_Layer/kernel

+3rd_Hidden_Layer/kernel/Read/ReadVariableOpReadVariableOp3rd_Hidden_Layer/kernel* 
_output_shapes
:
*
dtype0

3rd_Hidden_Layer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_name3rd_Hidden_Layer/bias
|
)3rd_Hidden_Layer/bias/Read/ReadVariableOpReadVariableOp3rd_Hidden_Layer/bias*
_output_shapes	
:*
dtype0

4th_Hidden_Layer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*(
shared_name4th_Hidden_Layer/kernel

+4th_Hidden_Layer/kernel/Read/ReadVariableOpReadVariableOp4th_Hidden_Layer/kernel* 
_output_shapes
:
*
dtype0

4th_Hidden_Layer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_name4th_Hidden_Layer/bias
|
)4th_Hidden_Layer/bias/Read/ReadVariableOpReadVariableOp4th_Hidden_Layer/bias*
_output_shapes	
:*
dtype0

5th_Hidden_Layer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *(
shared_name5th_Hidden_Layer/kernel

+5th_Hidden_Layer/kernel/Read/ReadVariableOpReadVariableOp5th_Hidden_Layer/kernel*
_output_shapes
:	 *
dtype0

5th_Hidden_Layer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_name5th_Hidden_Layer/bias
{
)5th_Hidden_Layer/bias/Read/ReadVariableOpReadVariableOp5th_Hidden_Layer/bias*
_output_shapes
: *
dtype0

Output_Layer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *$
shared_nameOutput_Layer/kernel
{
'Output_Layer/kernel/Read/ReadVariableOpReadVariableOpOutput_Layer/kernel*
_output_shapes

: *
dtype0
z
Output_Layer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameOutput_Layer/bias
s
%Output_Layer/bias/Read/ReadVariableOpReadVariableOpOutput_Layer/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
t
true_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nametrue_positives
m
"true_positives/Read/ReadVariableOpReadVariableOptrue_positives*
_output_shapes
:*
dtype0
v
false_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namefalse_positives
o
#false_positives/Read/ReadVariableOpReadVariableOpfalse_positives*
_output_shapes
:*
dtype0
v
false_negativesVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namefalse_negatives
o
#false_negatives/Read/ReadVariableOpReadVariableOpfalse_negatives*
_output_shapes
:*
dtype0

weights_intermediateVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameweights_intermediate
y
(weights_intermediate/Read/ReadVariableOpReadVariableOpweights_intermediate*
_output_shapes
:*
dtype0

Adam/1st_Hidden_Layer/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: */
shared_name Adam/1st_Hidden_Layer/kernel/m

2Adam/1st_Hidden_Layer/kernel/m/Read/ReadVariableOpReadVariableOpAdam/1st_Hidden_Layer/kernel/m*
_output_shapes

: *
dtype0

Adam/1st_Hidden_Layer/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_nameAdam/1st_Hidden_Layer/bias/m

0Adam/1st_Hidden_Layer/bias/m/Read/ReadVariableOpReadVariableOpAdam/1st_Hidden_Layer/bias/m*
_output_shapes
: *
dtype0

Adam/2nd_Hidden_Layer/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 */
shared_name Adam/2nd_Hidden_Layer/kernel/m

2Adam/2nd_Hidden_Layer/kernel/m/Read/ReadVariableOpReadVariableOpAdam/2nd_Hidden_Layer/kernel/m*
_output_shapes
:	 *
dtype0

Adam/2nd_Hidden_Layer/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameAdam/2nd_Hidden_Layer/bias/m

0Adam/2nd_Hidden_Layer/bias/m/Read/ReadVariableOpReadVariableOpAdam/2nd_Hidden_Layer/bias/m*
_output_shapes	
:*
dtype0

Adam/3rd_Hidden_Layer/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*/
shared_name Adam/3rd_Hidden_Layer/kernel/m

2Adam/3rd_Hidden_Layer/kernel/m/Read/ReadVariableOpReadVariableOpAdam/3rd_Hidden_Layer/kernel/m* 
_output_shapes
:
*
dtype0

Adam/3rd_Hidden_Layer/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameAdam/3rd_Hidden_Layer/bias/m

0Adam/3rd_Hidden_Layer/bias/m/Read/ReadVariableOpReadVariableOpAdam/3rd_Hidden_Layer/bias/m*
_output_shapes	
:*
dtype0

Adam/4th_Hidden_Layer/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*/
shared_name Adam/4th_Hidden_Layer/kernel/m

2Adam/4th_Hidden_Layer/kernel/m/Read/ReadVariableOpReadVariableOpAdam/4th_Hidden_Layer/kernel/m* 
_output_shapes
:
*
dtype0

Adam/4th_Hidden_Layer/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameAdam/4th_Hidden_Layer/bias/m

0Adam/4th_Hidden_Layer/bias/m/Read/ReadVariableOpReadVariableOpAdam/4th_Hidden_Layer/bias/m*
_output_shapes	
:*
dtype0

Adam/5th_Hidden_Layer/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 */
shared_name Adam/5th_Hidden_Layer/kernel/m

2Adam/5th_Hidden_Layer/kernel/m/Read/ReadVariableOpReadVariableOpAdam/5th_Hidden_Layer/kernel/m*
_output_shapes
:	 *
dtype0

Adam/5th_Hidden_Layer/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_nameAdam/5th_Hidden_Layer/bias/m

0Adam/5th_Hidden_Layer/bias/m/Read/ReadVariableOpReadVariableOpAdam/5th_Hidden_Layer/bias/m*
_output_shapes
: *
dtype0

Adam/Output_Layer/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *+
shared_nameAdam/Output_Layer/kernel/m

.Adam/Output_Layer/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Output_Layer/kernel/m*
_output_shapes

: *
dtype0

Adam/Output_Layer/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/Output_Layer/bias/m

,Adam/Output_Layer/bias/m/Read/ReadVariableOpReadVariableOpAdam/Output_Layer/bias/m*
_output_shapes
:*
dtype0

Adam/1st_Hidden_Layer/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: */
shared_name Adam/1st_Hidden_Layer/kernel/v

2Adam/1st_Hidden_Layer/kernel/v/Read/ReadVariableOpReadVariableOpAdam/1st_Hidden_Layer/kernel/v*
_output_shapes

: *
dtype0

Adam/1st_Hidden_Layer/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_nameAdam/1st_Hidden_Layer/bias/v

0Adam/1st_Hidden_Layer/bias/v/Read/ReadVariableOpReadVariableOpAdam/1st_Hidden_Layer/bias/v*
_output_shapes
: *
dtype0

Adam/2nd_Hidden_Layer/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 */
shared_name Adam/2nd_Hidden_Layer/kernel/v

2Adam/2nd_Hidden_Layer/kernel/v/Read/ReadVariableOpReadVariableOpAdam/2nd_Hidden_Layer/kernel/v*
_output_shapes
:	 *
dtype0

Adam/2nd_Hidden_Layer/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameAdam/2nd_Hidden_Layer/bias/v

0Adam/2nd_Hidden_Layer/bias/v/Read/ReadVariableOpReadVariableOpAdam/2nd_Hidden_Layer/bias/v*
_output_shapes	
:*
dtype0

Adam/3rd_Hidden_Layer/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*/
shared_name Adam/3rd_Hidden_Layer/kernel/v

2Adam/3rd_Hidden_Layer/kernel/v/Read/ReadVariableOpReadVariableOpAdam/3rd_Hidden_Layer/kernel/v* 
_output_shapes
:
*
dtype0

Adam/3rd_Hidden_Layer/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameAdam/3rd_Hidden_Layer/bias/v

0Adam/3rd_Hidden_Layer/bias/v/Read/ReadVariableOpReadVariableOpAdam/3rd_Hidden_Layer/bias/v*
_output_shapes	
:*
dtype0

Adam/4th_Hidden_Layer/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*/
shared_name Adam/4th_Hidden_Layer/kernel/v

2Adam/4th_Hidden_Layer/kernel/v/Read/ReadVariableOpReadVariableOpAdam/4th_Hidden_Layer/kernel/v* 
_output_shapes
:
*
dtype0

Adam/4th_Hidden_Layer/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameAdam/4th_Hidden_Layer/bias/v

0Adam/4th_Hidden_Layer/bias/v/Read/ReadVariableOpReadVariableOpAdam/4th_Hidden_Layer/bias/v*
_output_shapes	
:*
dtype0

Adam/5th_Hidden_Layer/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 */
shared_name Adam/5th_Hidden_Layer/kernel/v

2Adam/5th_Hidden_Layer/kernel/v/Read/ReadVariableOpReadVariableOpAdam/5th_Hidden_Layer/kernel/v*
_output_shapes
:	 *
dtype0

Adam/5th_Hidden_Layer/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_nameAdam/5th_Hidden_Layer/bias/v

0Adam/5th_Hidden_Layer/bias/v/Read/ReadVariableOpReadVariableOpAdam/5th_Hidden_Layer/bias/v*
_output_shapes
: *
dtype0

Adam/Output_Layer/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *+
shared_nameAdam/Output_Layer/kernel/v

.Adam/Output_Layer/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Output_Layer/kernel/v*
_output_shapes

: *
dtype0

Adam/Output_Layer/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/Output_Layer/bias/v

,Adam/Output_Layer/bias/v/Read/ReadVariableOpReadVariableOpAdam/Output_Layer/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
ÎQ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Q
valueÿPBüP BõP
¶
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
layer_with_weights-5
layer-5
	optimizer
	variables
	trainable_variables

regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
¦

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
¦

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
¦

 kernel
!bias
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses*
¦

(kernel
)bias
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses*
¦

0kernel
1bias
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses*
¦

8kernel
9bias
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses*
ª
@iter

Abeta_1

Bbeta_2
	Cdecay
Dlearning_ratemvmwmxmy mz!m{(m|)m}0m~1m8m9mvvvv v!v(v)v0v1v8v9v*
Z
0
1
2
3
 4
!5
(6
)7
08
19
810
911*
Z
0
1
2
3
 4
!5
(6
)7
08
19
810
911*
* 
°
Enon_trainable_variables

Flayers
Gmetrics
Hlayer_regularization_losses
Ilayer_metrics
	variables
	trainable_variables

regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

Jserving_default* 
ga
VARIABLE_VALUE1st_Hidden_Layer/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE1st_Hidden_Layer/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

Knon_trainable_variables

Llayers
Mmetrics
Nlayer_regularization_losses
Olayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
ga
VARIABLE_VALUE2nd_Hidden_Layer/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE2nd_Hidden_Layer/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

Pnon_trainable_variables

Qlayers
Rmetrics
Slayer_regularization_losses
Tlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
ga
VARIABLE_VALUE3rd_Hidden_Layer/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE3rd_Hidden_Layer/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

 0
!1*

 0
!1*
* 

Unon_trainable_variables

Vlayers
Wmetrics
Xlayer_regularization_losses
Ylayer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses*
* 
* 
ga
VARIABLE_VALUE4th_Hidden_Layer/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE4th_Hidden_Layer/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

(0
)1*

(0
)1*
* 

Znon_trainable_variables

[layers
\metrics
]layer_regularization_losses
^layer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses*
* 
* 
ga
VARIABLE_VALUE5th_Hidden_Layer/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE5th_Hidden_Layer/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*

00
11*

00
11*
* 

_non_trainable_variables

`layers
ametrics
blayer_regularization_losses
clayer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses*
* 
* 
c]
VARIABLE_VALUEOutput_Layer/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEOutput_Layer/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*

80
91*

80
91*
* 

dnon_trainable_variables

elayers
fmetrics
glayer_regularization_losses
hlayer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses*
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
.
0
1
2
3
4
5*

i0
j1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
8
	ktotal
	lcount
m	variables
n	keras_api*

o
init_shape
ptrue_positives
qfalse_positives
rfalse_negatives
sweights_intermediate
t	variables
u	keras_api*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

k0
l1*

m	variables*
* 
e_
VARIABLE_VALUEtrue_positives=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEfalse_positives>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEfalse_negatives>keras_api/metrics/1/false_negatives/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEweights_intermediateCkeras_api/metrics/1/weights_intermediate/.ATTRIBUTES/VARIABLE_VALUE*
 
p0
q1
r2
s3*

t	variables*

VARIABLE_VALUEAdam/1st_Hidden_Layer/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/1st_Hidden_Layer/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/2nd_Hidden_Layer/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/2nd_Hidden_Layer/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/3rd_Hidden_Layer/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/3rd_Hidden_Layer/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/4th_Hidden_Layer/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/4th_Hidden_Layer/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/5th_Hidden_Layer/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/5th_Hidden_Layer/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/Output_Layer/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/Output_Layer/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/1st_Hidden_Layer/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/1st_Hidden_Layer/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/2nd_Hidden_Layer/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/2nd_Hidden_Layer/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/3rd_Hidden_Layer/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/3rd_Hidden_Layer/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/4th_Hidden_Layer/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/4th_Hidden_Layer/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/5th_Hidden_Layer/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/5th_Hidden_Layer/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/Output_Layer/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/Output_Layer/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
serving_default_input_1Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
ä
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_11st_Hidden_Layer/kernel1st_Hidden_Layer/bias2nd_Hidden_Layer/kernel2nd_Hidden_Layer/bias3rd_Hidden_Layer/kernel3rd_Hidden_Layer/bias4th_Hidden_Layer/kernel4th_Hidden_Layer/bias5th_Hidden_Layer/kernel5th_Hidden_Layer/biasOutput_Layer/kernelOutput_Layer/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *,
f'R%
#__inference_signature_wrapper_12097
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename+1st_Hidden_Layer/kernel/Read/ReadVariableOp)1st_Hidden_Layer/bias/Read/ReadVariableOp+2nd_Hidden_Layer/kernel/Read/ReadVariableOp)2nd_Hidden_Layer/bias/Read/ReadVariableOp+3rd_Hidden_Layer/kernel/Read/ReadVariableOp)3rd_Hidden_Layer/bias/Read/ReadVariableOp+4th_Hidden_Layer/kernel/Read/ReadVariableOp)4th_Hidden_Layer/bias/Read/ReadVariableOp+5th_Hidden_Layer/kernel/Read/ReadVariableOp)5th_Hidden_Layer/bias/Read/ReadVariableOp'Output_Layer/kernel/Read/ReadVariableOp%Output_Layer/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp"true_positives/Read/ReadVariableOp#false_positives/Read/ReadVariableOp#false_negatives/Read/ReadVariableOp(weights_intermediate/Read/ReadVariableOp2Adam/1st_Hidden_Layer/kernel/m/Read/ReadVariableOp0Adam/1st_Hidden_Layer/bias/m/Read/ReadVariableOp2Adam/2nd_Hidden_Layer/kernel/m/Read/ReadVariableOp0Adam/2nd_Hidden_Layer/bias/m/Read/ReadVariableOp2Adam/3rd_Hidden_Layer/kernel/m/Read/ReadVariableOp0Adam/3rd_Hidden_Layer/bias/m/Read/ReadVariableOp2Adam/4th_Hidden_Layer/kernel/m/Read/ReadVariableOp0Adam/4th_Hidden_Layer/bias/m/Read/ReadVariableOp2Adam/5th_Hidden_Layer/kernel/m/Read/ReadVariableOp0Adam/5th_Hidden_Layer/bias/m/Read/ReadVariableOp.Adam/Output_Layer/kernel/m/Read/ReadVariableOp,Adam/Output_Layer/bias/m/Read/ReadVariableOp2Adam/1st_Hidden_Layer/kernel/v/Read/ReadVariableOp0Adam/1st_Hidden_Layer/bias/v/Read/ReadVariableOp2Adam/2nd_Hidden_Layer/kernel/v/Read/ReadVariableOp0Adam/2nd_Hidden_Layer/bias/v/Read/ReadVariableOp2Adam/3rd_Hidden_Layer/kernel/v/Read/ReadVariableOp0Adam/3rd_Hidden_Layer/bias/v/Read/ReadVariableOp2Adam/4th_Hidden_Layer/kernel/v/Read/ReadVariableOp0Adam/4th_Hidden_Layer/bias/v/Read/ReadVariableOp2Adam/5th_Hidden_Layer/kernel/v/Read/ReadVariableOp0Adam/5th_Hidden_Layer/bias/v/Read/ReadVariableOp.Adam/Output_Layer/kernel/v/Read/ReadVariableOp,Adam/Output_Layer/bias/v/Read/ReadVariableOpConst*<
Tin5
321	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *'
f"R 
__inference__traced_save_12381
Ò
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename1st_Hidden_Layer/kernel1st_Hidden_Layer/bias2nd_Hidden_Layer/kernel2nd_Hidden_Layer/bias3rd_Hidden_Layer/kernel3rd_Hidden_Layer/bias4th_Hidden_Layer/kernel4th_Hidden_Layer/bias5th_Hidden_Layer/kernel5th_Hidden_Layer/biasOutput_Layer/kernelOutput_Layer/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttrue_positivesfalse_positivesfalse_negativesweights_intermediateAdam/1st_Hidden_Layer/kernel/mAdam/1st_Hidden_Layer/bias/mAdam/2nd_Hidden_Layer/kernel/mAdam/2nd_Hidden_Layer/bias/mAdam/3rd_Hidden_Layer/kernel/mAdam/3rd_Hidden_Layer/bias/mAdam/4th_Hidden_Layer/kernel/mAdam/4th_Hidden_Layer/bias/mAdam/5th_Hidden_Layer/kernel/mAdam/5th_Hidden_Layer/bias/mAdam/Output_Layer/kernel/mAdam/Output_Layer/bias/mAdam/1st_Hidden_Layer/kernel/vAdam/1st_Hidden_Layer/bias/vAdam/2nd_Hidden_Layer/kernel/vAdam/2nd_Hidden_Layer/bias/vAdam/3rd_Hidden_Layer/kernel/vAdam/3rd_Hidden_Layer/bias/vAdam/4th_Hidden_Layer/kernel/vAdam/4th_Hidden_Layer/bias/vAdam/5th_Hidden_Layer/kernel/vAdam/5th_Hidden_Layer/bias/vAdam/Output_Layer/kernel/vAdam/Output_Layer/bias/v*;
Tin4
220*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__traced_restore_12532ÐÔ
Í

©
#__inference_signature_wrapper_12097
input_1
unknown: 
	unknown_0: 
	unknown_1:	 
	unknown_2:	
	unknown_3:

	unknown_4:	
	unknown_5:

	unknown_6:	
	unknown_7:	 
	unknown_8: 
	unknown_9: 

unknown_10:
identity¢StatefulPartitionedCall¼
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *)
f$R"
 __inference__wrapped_model_11524o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
ù

°
*__inference_sequential_layer_call_fn_11661
input_1
unknown: 
	unknown_0: 
	unknown_1:	 
	unknown_2:	
	unknown_3:

	unknown_4:	
	unknown_5:

	unknown_6:	
	unknown_7:	 
	unknown_8: 
	unknown_9: 

unknown_10:
identity¢StatefulPartitionedCallá
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_11634o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
Ú
 
0__inference_3rd_Hidden_Layer_layer_call_fn_12146

inputs
unknown:

	unknown_0:	
identity¢StatefulPartitionedCallä
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_3rd_Hidden_Layer_layer_call_and_return_conditional_losses_11576p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¦

ý
K__inference_5th_Hidden_Layer_layer_call_and_return_conditional_losses_12197

inputs1
matmul_readvariableop_resource:	 -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	 *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¢

ü
K__inference_1st_Hidden_Layer_layer_call_and_return_conditional_losses_12117

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ö

¯
*__inference_sequential_layer_call_fn_11945

inputs
unknown: 
	unknown_0: 
	unknown_1:	 
	unknown_2:	
	unknown_3:

	unknown_4:	
	unknown_5:

	unknown_6:	
	unknown_7:	 
	unknown_8: 
	unknown_9: 

unknown_10:
identity¢StatefulPartitionedCallà
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_11634o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
$
º
E__inference_sequential_layer_call_and_return_conditional_losses_11910
input_1'
st_hidden_layer_11879: #
st_hidden_layer_11881: (
nd_hidden_layer_11884:	 $
nd_hidden_layer_11886:	)
rd_hidden_layer_11889:
$
rd_hidden_layer_11891:	)
th_hidden_layer_11894:
$
th_hidden_layer_11896:	(
th_hidden_layer_11899:	 #
th_hidden_layer_11901: $
output_layer_11904:  
output_layer_11906:
identity¢(1st_Hidden_Layer/StatefulPartitionedCall¢(2nd_Hidden_Layer/StatefulPartitionedCall¢(3rd_Hidden_Layer/StatefulPartitionedCall¢(4th_Hidden_Layer/StatefulPartitionedCall¢(5th_Hidden_Layer/StatefulPartitionedCall¢$Output_Layer/StatefulPartitionedCall
(1st_Hidden_Layer/StatefulPartitionedCallStatefulPartitionedCallinput_1st_hidden_layer_11879st_hidden_layer_11881*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_1st_Hidden_Layer_layer_call_and_return_conditional_losses_11542º
(2nd_Hidden_Layer/StatefulPartitionedCallStatefulPartitionedCall11st_Hidden_Layer/StatefulPartitionedCall:output:0nd_hidden_layer_11884nd_hidden_layer_11886*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_2nd_Hidden_Layer_layer_call_and_return_conditional_losses_11559º
(3rd_Hidden_Layer/StatefulPartitionedCallStatefulPartitionedCall12nd_Hidden_Layer/StatefulPartitionedCall:output:0rd_hidden_layer_11889rd_hidden_layer_11891*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_3rd_Hidden_Layer_layer_call_and_return_conditional_losses_11576º
(4th_Hidden_Layer/StatefulPartitionedCallStatefulPartitionedCall13rd_Hidden_Layer/StatefulPartitionedCall:output:0th_hidden_layer_11894th_hidden_layer_11896*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_4th_Hidden_Layer_layer_call_and_return_conditional_losses_11593¹
(5th_Hidden_Layer/StatefulPartitionedCallStatefulPartitionedCall14th_Hidden_Layer/StatefulPartitionedCall:output:0th_hidden_layer_11899th_hidden_layer_11901*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_5th_Hidden_Layer_layer_call_and_return_conditional_losses_11610«
$Output_Layer/StatefulPartitionedCallStatefulPartitionedCall15th_Hidden_Layer/StatefulPartitionedCall:output:0output_layer_11904output_layer_11906*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_Output_Layer_layer_call_and_return_conditional_losses_11627|
IdentityIdentity-Output_Layer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
NoOpNoOp)^1st_Hidden_Layer/StatefulPartitionedCall)^2nd_Hidden_Layer/StatefulPartitionedCall)^3rd_Hidden_Layer/StatefulPartitionedCall)^4th_Hidden_Layer/StatefulPartitionedCall)^5th_Hidden_Layer/StatefulPartitionedCall%^Output_Layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 2T
(1st_Hidden_Layer/StatefulPartitionedCall(1st_Hidden_Layer/StatefulPartitionedCall2T
(2nd_Hidden_Layer/StatefulPartitionedCall(2nd_Hidden_Layer/StatefulPartitionedCall2T
(3rd_Hidden_Layer/StatefulPartitionedCall(3rd_Hidden_Layer/StatefulPartitionedCall2T
(4th_Hidden_Layer/StatefulPartitionedCall(4th_Hidden_Layer/StatefulPartitionedCall2T
(5th_Hidden_Layer/StatefulPartitionedCall(5th_Hidden_Layer/StatefulPartitionedCall2L
$Output_Layer/StatefulPartitionedCall$Output_Layer/StatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
×

0__inference_2nd_Hidden_Layer_layer_call_fn_12126

inputs
unknown:	 
	unknown_0:	
identity¢StatefulPartitionedCallä
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_2nd_Hidden_Layer_layer_call_and_return_conditional_losses_11559p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ë

,__inference_Output_Layer_layer_call_fn_12206

inputs
unknown: 
	unknown_0:
identity¢StatefulPartitionedCallß
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_Output_Layer_layer_call_and_return_conditional_losses_11627o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
»¿
Ê
!__inference__traced_restore_12532
file_prefix:
(assignvariableop_1st_hidden_layer_kernel: 6
(assignvariableop_1_1st_hidden_layer_bias: =
*assignvariableop_2_2nd_hidden_layer_kernel:	 7
(assignvariableop_3_2nd_hidden_layer_bias:	>
*assignvariableop_4_3rd_hidden_layer_kernel:
7
(assignvariableop_5_3rd_hidden_layer_bias:	>
*assignvariableop_6_4th_hidden_layer_kernel:
7
(assignvariableop_7_4th_hidden_layer_bias:	=
*assignvariableop_8_5th_hidden_layer_kernel:	 6
(assignvariableop_9_5th_hidden_layer_bias: 9
'assignvariableop_10_output_layer_kernel: 3
%assignvariableop_11_output_layer_bias:'
assignvariableop_12_adam_iter:	 )
assignvariableop_13_adam_beta_1: )
assignvariableop_14_adam_beta_2: (
assignvariableop_15_adam_decay: 0
&assignvariableop_16_adam_learning_rate: #
assignvariableop_17_total: #
assignvariableop_18_count: 0
"assignvariableop_19_true_positives:1
#assignvariableop_20_false_positives:1
#assignvariableop_21_false_negatives:6
(assignvariableop_22_weights_intermediate:D
2assignvariableop_23_adam_1st_hidden_layer_kernel_m: >
0assignvariableop_24_adam_1st_hidden_layer_bias_m: E
2assignvariableop_25_adam_2nd_hidden_layer_kernel_m:	 ?
0assignvariableop_26_adam_2nd_hidden_layer_bias_m:	F
2assignvariableop_27_adam_3rd_hidden_layer_kernel_m:
?
0assignvariableop_28_adam_3rd_hidden_layer_bias_m:	F
2assignvariableop_29_adam_4th_hidden_layer_kernel_m:
?
0assignvariableop_30_adam_4th_hidden_layer_bias_m:	E
2assignvariableop_31_adam_5th_hidden_layer_kernel_m:	 >
0assignvariableop_32_adam_5th_hidden_layer_bias_m: @
.assignvariableop_33_adam_output_layer_kernel_m: :
,assignvariableop_34_adam_output_layer_bias_m:D
2assignvariableop_35_adam_1st_hidden_layer_kernel_v: >
0assignvariableop_36_adam_1st_hidden_layer_bias_v: E
2assignvariableop_37_adam_2nd_hidden_layer_kernel_v:	 ?
0assignvariableop_38_adam_2nd_hidden_layer_bias_v:	F
2assignvariableop_39_adam_3rd_hidden_layer_kernel_v:
?
0assignvariableop_40_adam_3rd_hidden_layer_bias_v:	F
2assignvariableop_41_adam_4th_hidden_layer_kernel_v:
?
0assignvariableop_42_adam_4th_hidden_layer_bias_v:	E
2assignvariableop_43_adam_5th_hidden_layer_kernel_v:	 >
0assignvariableop_44_adam_5th_hidden_layer_bias_v: @
.assignvariableop_45_adam_output_layer_kernel_v: :
,assignvariableop_46_adam_output_layer_bias_v:
identity_48¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9¾
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:0*
dtype0*ä
valueÚB×0B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBCkeras_api/metrics/1/weights_intermediate/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHÐ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:0*
dtype0*s
valuejBh0B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Ö
_output_shapesÃ
À::::::::::::::::::::::::::::::::::::::::::::::::*>
dtypes4
220	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp(assignvariableop_1st_hidden_layer_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp(assignvariableop_1_1st_hidden_layer_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp*assignvariableop_2_2nd_hidden_layer_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp(assignvariableop_3_2nd_hidden_layer_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp*assignvariableop_4_3rd_hidden_layer_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp(assignvariableop_5_3rd_hidden_layer_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp*assignvariableop_6_4th_hidden_layer_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp(assignvariableop_7_4th_hidden_layer_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp*assignvariableop_8_5th_hidden_layer_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp(assignvariableop_9_5th_hidden_layer_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp'assignvariableop_10_output_layer_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOp%assignvariableop_11_output_layer_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_iterIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_beta_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_beta_2Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOpassignvariableop_15_adam_decayIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp&assignvariableop_16_adam_learning_rateIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOpassignvariableop_17_totalIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOpassignvariableop_18_countIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp"assignvariableop_19_true_positivesIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp#assignvariableop_20_false_positivesIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp#assignvariableop_21_false_negativesIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp(assignvariableop_22_weights_intermediateIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_23AssignVariableOp2assignvariableop_23_adam_1st_hidden_layer_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_24AssignVariableOp0assignvariableop_24_adam_1st_hidden_layer_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_25AssignVariableOp2assignvariableop_25_adam_2nd_hidden_layer_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_26AssignVariableOp0assignvariableop_26_adam_2nd_hidden_layer_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_27AssignVariableOp2assignvariableop_27_adam_3rd_hidden_layer_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_28AssignVariableOp0assignvariableop_28_adam_3rd_hidden_layer_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_29AssignVariableOp2assignvariableop_29_adam_4th_hidden_layer_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_30AssignVariableOp0assignvariableop_30_adam_4th_hidden_layer_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_31AssignVariableOp2assignvariableop_31_adam_5th_hidden_layer_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_32AssignVariableOp0assignvariableop_32_adam_5th_hidden_layer_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_33AssignVariableOp.assignvariableop_33_adam_output_layer_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_34AssignVariableOp,assignvariableop_34_adam_output_layer_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_35AssignVariableOp2assignvariableop_35_adam_1st_hidden_layer_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_36AssignVariableOp0assignvariableop_36_adam_1st_hidden_layer_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_37AssignVariableOp2assignvariableop_37_adam_2nd_hidden_layer_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_38AssignVariableOp0assignvariableop_38_adam_2nd_hidden_layer_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_39AssignVariableOp2assignvariableop_39_adam_3rd_hidden_layer_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_40AssignVariableOp0assignvariableop_40_adam_3rd_hidden_layer_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_41AssignVariableOp2assignvariableop_41_adam_4th_hidden_layer_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_42AssignVariableOp0assignvariableop_42_adam_4th_hidden_layer_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_43AssignVariableOp2assignvariableop_43_adam_5th_hidden_layer_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_44AssignVariableOp0assignvariableop_44_adam_5th_hidden_layer_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_45AssignVariableOp.assignvariableop_45_adam_output_layer_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_46AssignVariableOp,assignvariableop_46_adam_output_layer_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 Ù
Identity_47Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_48IdentityIdentity_47:output:0^NoOp_1*
T0*
_output_shapes
: Æ
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_48Identity_48:output:0*s
_input_shapesb
`: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
¦

ý
K__inference_5th_Hidden_Layer_layer_call_and_return_conditional_losses_11610

inputs1
matmul_readvariableop_resource:	 -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	 *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


ø
G__inference_Output_Layer_layer_call_and_return_conditional_losses_12217

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
®

ÿ
K__inference_3rd_Hidden_Layer_layer_call_and_return_conditional_losses_11576

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
®

ÿ
K__inference_4th_Hidden_Layer_layer_call_and_return_conditional_losses_11593

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ö

0__inference_5th_Hidden_Layer_layer_call_fn_12186

inputs
unknown:	 
	unknown_0: 
identity¢StatefulPartitionedCallã
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_5th_Hidden_Layer_layer_call_and_return_conditional_losses_11610o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
$
¹
E__inference_sequential_layer_call_and_return_conditional_losses_11634

inputs'
st_hidden_layer_11543: #
st_hidden_layer_11545: (
nd_hidden_layer_11560:	 $
nd_hidden_layer_11562:	)
rd_hidden_layer_11577:
$
rd_hidden_layer_11579:	)
th_hidden_layer_11594:
$
th_hidden_layer_11596:	(
th_hidden_layer_11611:	 #
th_hidden_layer_11613: $
output_layer_11628:  
output_layer_11630:
identity¢(1st_Hidden_Layer/StatefulPartitionedCall¢(2nd_Hidden_Layer/StatefulPartitionedCall¢(3rd_Hidden_Layer/StatefulPartitionedCall¢(4th_Hidden_Layer/StatefulPartitionedCall¢(5th_Hidden_Layer/StatefulPartitionedCall¢$Output_Layer/StatefulPartitionedCall
(1st_Hidden_Layer/StatefulPartitionedCallStatefulPartitionedCallinputsst_hidden_layer_11543st_hidden_layer_11545*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_1st_Hidden_Layer_layer_call_and_return_conditional_losses_11542º
(2nd_Hidden_Layer/StatefulPartitionedCallStatefulPartitionedCall11st_Hidden_Layer/StatefulPartitionedCall:output:0nd_hidden_layer_11560nd_hidden_layer_11562*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_2nd_Hidden_Layer_layer_call_and_return_conditional_losses_11559º
(3rd_Hidden_Layer/StatefulPartitionedCallStatefulPartitionedCall12nd_Hidden_Layer/StatefulPartitionedCall:output:0rd_hidden_layer_11577rd_hidden_layer_11579*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_3rd_Hidden_Layer_layer_call_and_return_conditional_losses_11576º
(4th_Hidden_Layer/StatefulPartitionedCallStatefulPartitionedCall13rd_Hidden_Layer/StatefulPartitionedCall:output:0th_hidden_layer_11594th_hidden_layer_11596*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_4th_Hidden_Layer_layer_call_and_return_conditional_losses_11593¹
(5th_Hidden_Layer/StatefulPartitionedCallStatefulPartitionedCall14th_Hidden_Layer/StatefulPartitionedCall:output:0th_hidden_layer_11611th_hidden_layer_11613*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_5th_Hidden_Layer_layer_call_and_return_conditional_losses_11610«
$Output_Layer/StatefulPartitionedCallStatefulPartitionedCall15th_Hidden_Layer/StatefulPartitionedCall:output:0output_layer_11628output_layer_11630*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_Output_Layer_layer_call_and_return_conditional_losses_11627|
IdentityIdentity-Output_Layer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
NoOpNoOp)^1st_Hidden_Layer/StatefulPartitionedCall)^2nd_Hidden_Layer/StatefulPartitionedCall)^3rd_Hidden_Layer/StatefulPartitionedCall)^4th_Hidden_Layer/StatefulPartitionedCall)^5th_Hidden_Layer/StatefulPartitionedCall%^Output_Layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 2T
(1st_Hidden_Layer/StatefulPartitionedCall(1st_Hidden_Layer/StatefulPartitionedCall2T
(2nd_Hidden_Layer/StatefulPartitionedCall(2nd_Hidden_Layer/StatefulPartitionedCall2T
(3rd_Hidden_Layer/StatefulPartitionedCall(3rd_Hidden_Layer/StatefulPartitionedCall2T
(4th_Hidden_Layer/StatefulPartitionedCall(4th_Hidden_Layer/StatefulPartitionedCall2T
(5th_Hidden_Layer/StatefulPartitionedCall(5th_Hidden_Layer/StatefulPartitionedCall2L
$Output_Layer/StatefulPartitionedCall$Output_Layer/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ó<
Û

E__inference_sequential_layer_call_and_return_conditional_losses_12020

inputs@
.st_hidden_layer_matmul_readvariableop_resource: =
/st_hidden_layer_biasadd_readvariableop_resource: A
.nd_hidden_layer_matmul_readvariableop_resource:	 >
/nd_hidden_layer_biasadd_readvariableop_resource:	B
.rd_hidden_layer_matmul_readvariableop_resource:
>
/rd_hidden_layer_biasadd_readvariableop_resource:	B
.th_hidden_layer_matmul_readvariableop_resource:
>
/th_hidden_layer_biasadd_readvariableop_resource:	C
0th_hidden_layer_matmul_readvariableop_resource_0:	 ?
1th_hidden_layer_biasadd_readvariableop_resource_0: =
+output_layer_matmul_readvariableop_resource: :
,output_layer_biasadd_readvariableop_resource:
identity¢'1st_Hidden_Layer/BiasAdd/ReadVariableOp¢&1st_Hidden_Layer/MatMul/ReadVariableOp¢'2nd_Hidden_Layer/BiasAdd/ReadVariableOp¢&2nd_Hidden_Layer/MatMul/ReadVariableOp¢'3rd_Hidden_Layer/BiasAdd/ReadVariableOp¢&3rd_Hidden_Layer/MatMul/ReadVariableOp¢'4th_Hidden_Layer/BiasAdd/ReadVariableOp¢&4th_Hidden_Layer/MatMul/ReadVariableOp¢'5th_Hidden_Layer/BiasAdd/ReadVariableOp¢&5th_Hidden_Layer/MatMul/ReadVariableOp¢#Output_Layer/BiasAdd/ReadVariableOp¢"Output_Layer/MatMul/ReadVariableOp
&1st_Hidden_Layer/MatMul/ReadVariableOpReadVariableOp.st_hidden_layer_matmul_readvariableop_resource*
_output_shapes

: *
dtype0
1st_Hidden_Layer/MatMulMatMulinputs.1st_Hidden_Layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
'1st_Hidden_Layer/BiasAdd/ReadVariableOpReadVariableOp/st_hidden_layer_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0©
1st_Hidden_Layer/BiasAddBiasAdd!1st_Hidden_Layer/MatMul:product:0/1st_Hidden_Layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
1st_Hidden_Layer/ReluRelu!1st_Hidden_Layer/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
&2nd_Hidden_Layer/MatMul/ReadVariableOpReadVariableOp.nd_hidden_layer_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype0©
2nd_Hidden_Layer/MatMulMatMul#1st_Hidden_Layer/Relu:activations:0.2nd_Hidden_Layer/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'2nd_Hidden_Layer/BiasAdd/ReadVariableOpReadVariableOp/nd_hidden_layer_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ª
2nd_Hidden_Layer/BiasAddBiasAdd!2nd_Hidden_Layer/MatMul:product:0/2nd_Hidden_Layer/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
2nd_Hidden_Layer/ReluRelu!2nd_Hidden_Layer/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&3rd_Hidden_Layer/MatMul/ReadVariableOpReadVariableOp.rd_hidden_layer_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0©
3rd_Hidden_Layer/MatMulMatMul#2nd_Hidden_Layer/Relu:activations:0.3rd_Hidden_Layer/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'3rd_Hidden_Layer/BiasAdd/ReadVariableOpReadVariableOp/rd_hidden_layer_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ª
3rd_Hidden_Layer/BiasAddBiasAdd!3rd_Hidden_Layer/MatMul:product:0/3rd_Hidden_Layer/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
3rd_Hidden_Layer/ReluRelu!3rd_Hidden_Layer/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&4th_Hidden_Layer/MatMul/ReadVariableOpReadVariableOp.th_hidden_layer_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0©
4th_Hidden_Layer/MatMulMatMul#3rd_Hidden_Layer/Relu:activations:0.4th_Hidden_Layer/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'4th_Hidden_Layer/BiasAdd/ReadVariableOpReadVariableOp/th_hidden_layer_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ª
4th_Hidden_Layer/BiasAddBiasAdd!4th_Hidden_Layer/MatMul:product:0/4th_Hidden_Layer/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
4th_Hidden_Layer/ReluRelu!4th_Hidden_Layer/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&5th_Hidden_Layer/MatMul/ReadVariableOpReadVariableOp0th_hidden_layer_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype0¨
5th_Hidden_Layer/MatMulMatMul#4th_Hidden_Layer/Relu:activations:0.5th_Hidden_Layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
'5th_Hidden_Layer/BiasAdd/ReadVariableOpReadVariableOp1th_hidden_layer_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0©
5th_Hidden_Layer/BiasAddBiasAdd!5th_Hidden_Layer/MatMul:product:0/5th_Hidden_Layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
5th_Hidden_Layer/ReluRelu!5th_Hidden_Layer/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"Output_Layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

: *
dtype0 
Output_Layer/MatMulMatMul#5th_Hidden_Layer/Relu:activations:0*Output_Layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#Output_Layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
Output_Layer/BiasAddBiasAddOutput_Layer/MatMul:product:0+Output_Layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
Output_Layer/SigmoidSigmoidOutput_Layer/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
IdentityIdentityOutput_Layer/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
NoOpNoOp(^1st_Hidden_Layer/BiasAdd/ReadVariableOp'^1st_Hidden_Layer/MatMul/ReadVariableOp(^2nd_Hidden_Layer/BiasAdd/ReadVariableOp'^2nd_Hidden_Layer/MatMul/ReadVariableOp(^3rd_Hidden_Layer/BiasAdd/ReadVariableOp'^3rd_Hidden_Layer/MatMul/ReadVariableOp(^4th_Hidden_Layer/BiasAdd/ReadVariableOp'^4th_Hidden_Layer/MatMul/ReadVariableOp(^5th_Hidden_Layer/BiasAdd/ReadVariableOp'^5th_Hidden_Layer/MatMul/ReadVariableOp$^Output_Layer/BiasAdd/ReadVariableOp#^Output_Layer/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 2R
'1st_Hidden_Layer/BiasAdd/ReadVariableOp'1st_Hidden_Layer/BiasAdd/ReadVariableOp2P
&1st_Hidden_Layer/MatMul/ReadVariableOp&1st_Hidden_Layer/MatMul/ReadVariableOp2R
'2nd_Hidden_Layer/BiasAdd/ReadVariableOp'2nd_Hidden_Layer/BiasAdd/ReadVariableOp2P
&2nd_Hidden_Layer/MatMul/ReadVariableOp&2nd_Hidden_Layer/MatMul/ReadVariableOp2R
'3rd_Hidden_Layer/BiasAdd/ReadVariableOp'3rd_Hidden_Layer/BiasAdd/ReadVariableOp2P
&3rd_Hidden_Layer/MatMul/ReadVariableOp&3rd_Hidden_Layer/MatMul/ReadVariableOp2R
'4th_Hidden_Layer/BiasAdd/ReadVariableOp'4th_Hidden_Layer/BiasAdd/ReadVariableOp2P
&4th_Hidden_Layer/MatMul/ReadVariableOp&4th_Hidden_Layer/MatMul/ReadVariableOp2R
'5th_Hidden_Layer/BiasAdd/ReadVariableOp'5th_Hidden_Layer/BiasAdd/ReadVariableOp2P
&5th_Hidden_Layer/MatMul/ReadVariableOp&5th_Hidden_Layer/MatMul/ReadVariableOp2J
#Output_Layer/BiasAdd/ReadVariableOp#Output_Layer/BiasAdd/ReadVariableOp2H
"Output_Layer/MatMul/ReadVariableOp"Output_Layer/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
õb
§
__inference__traced_save_12381
file_prefix6
2savev2_1st_hidden_layer_kernel_read_readvariableop4
0savev2_1st_hidden_layer_bias_read_readvariableop6
2savev2_2nd_hidden_layer_kernel_read_readvariableop4
0savev2_2nd_hidden_layer_bias_read_readvariableop6
2savev2_3rd_hidden_layer_kernel_read_readvariableop4
0savev2_3rd_hidden_layer_bias_read_readvariableop6
2savev2_4th_hidden_layer_kernel_read_readvariableop4
0savev2_4th_hidden_layer_bias_read_readvariableop6
2savev2_5th_hidden_layer_kernel_read_readvariableop4
0savev2_5th_hidden_layer_bias_read_readvariableop2
.savev2_output_layer_kernel_read_readvariableop0
,savev2_output_layer_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop-
)savev2_true_positives_read_readvariableop.
*savev2_false_positives_read_readvariableop.
*savev2_false_negatives_read_readvariableop3
/savev2_weights_intermediate_read_readvariableop=
9savev2_adam_1st_hidden_layer_kernel_m_read_readvariableop;
7savev2_adam_1st_hidden_layer_bias_m_read_readvariableop=
9savev2_adam_2nd_hidden_layer_kernel_m_read_readvariableop;
7savev2_adam_2nd_hidden_layer_bias_m_read_readvariableop=
9savev2_adam_3rd_hidden_layer_kernel_m_read_readvariableop;
7savev2_adam_3rd_hidden_layer_bias_m_read_readvariableop=
9savev2_adam_4th_hidden_layer_kernel_m_read_readvariableop;
7savev2_adam_4th_hidden_layer_bias_m_read_readvariableop=
9savev2_adam_5th_hidden_layer_kernel_m_read_readvariableop;
7savev2_adam_5th_hidden_layer_bias_m_read_readvariableop9
5savev2_adam_output_layer_kernel_m_read_readvariableop7
3savev2_adam_output_layer_bias_m_read_readvariableop=
9savev2_adam_1st_hidden_layer_kernel_v_read_readvariableop;
7savev2_adam_1st_hidden_layer_bias_v_read_readvariableop=
9savev2_adam_2nd_hidden_layer_kernel_v_read_readvariableop;
7savev2_adam_2nd_hidden_layer_bias_v_read_readvariableop=
9savev2_adam_3rd_hidden_layer_kernel_v_read_readvariableop;
7savev2_adam_3rd_hidden_layer_bias_v_read_readvariableop=
9savev2_adam_4th_hidden_layer_kernel_v_read_readvariableop;
7savev2_adam_4th_hidden_layer_bias_v_read_readvariableop=
9savev2_adam_5th_hidden_layer_kernel_v_read_readvariableop;
7savev2_adam_5th_hidden_layer_bias_v_read_readvariableop9
5savev2_adam_output_layer_kernel_v_read_readvariableop7
3savev2_adam_output_layer_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: »
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:0*
dtype0*ä
valueÚB×0B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBCkeras_api/metrics/1/weights_intermediate/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHÍ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:0*
dtype0*s
valuejBh0B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ß
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:02savev2_1st_hidden_layer_kernel_read_readvariableop0savev2_1st_hidden_layer_bias_read_readvariableop2savev2_2nd_hidden_layer_kernel_read_readvariableop0savev2_2nd_hidden_layer_bias_read_readvariableop2savev2_3rd_hidden_layer_kernel_read_readvariableop0savev2_3rd_hidden_layer_bias_read_readvariableop2savev2_4th_hidden_layer_kernel_read_readvariableop0savev2_4th_hidden_layer_bias_read_readvariableop2savev2_5th_hidden_layer_kernel_read_readvariableop0savev2_5th_hidden_layer_bias_read_readvariableop.savev2_output_layer_kernel_read_readvariableop,savev2_output_layer_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop)savev2_true_positives_read_readvariableop*savev2_false_positives_read_readvariableop*savev2_false_negatives_read_readvariableop/savev2_weights_intermediate_read_readvariableop9savev2_adam_1st_hidden_layer_kernel_m_read_readvariableop7savev2_adam_1st_hidden_layer_bias_m_read_readvariableop9savev2_adam_2nd_hidden_layer_kernel_m_read_readvariableop7savev2_adam_2nd_hidden_layer_bias_m_read_readvariableop9savev2_adam_3rd_hidden_layer_kernel_m_read_readvariableop7savev2_adam_3rd_hidden_layer_bias_m_read_readvariableop9savev2_adam_4th_hidden_layer_kernel_m_read_readvariableop7savev2_adam_4th_hidden_layer_bias_m_read_readvariableop9savev2_adam_5th_hidden_layer_kernel_m_read_readvariableop7savev2_adam_5th_hidden_layer_bias_m_read_readvariableop5savev2_adam_output_layer_kernel_m_read_readvariableop3savev2_adam_output_layer_bias_m_read_readvariableop9savev2_adam_1st_hidden_layer_kernel_v_read_readvariableop7savev2_adam_1st_hidden_layer_bias_v_read_readvariableop9savev2_adam_2nd_hidden_layer_kernel_v_read_readvariableop7savev2_adam_2nd_hidden_layer_bias_v_read_readvariableop9savev2_adam_3rd_hidden_layer_kernel_v_read_readvariableop7savev2_adam_3rd_hidden_layer_bias_v_read_readvariableop9savev2_adam_4th_hidden_layer_kernel_v_read_readvariableop7savev2_adam_4th_hidden_layer_bias_v_read_readvariableop9savev2_adam_5th_hidden_layer_kernel_v_read_readvariableop7savev2_adam_5th_hidden_layer_bias_v_read_readvariableop5savev2_adam_output_layer_kernel_v_read_readvariableop3savev2_adam_output_layer_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *>
dtypes4
220	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*ú
_input_shapesè
å: : : :	 ::
::
::	 : : :: : : : : : : ::::: : :	 ::
::
::	 : : :: : :	 ::
::
::	 : : :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

: : 

_output_shapes
: :%!

_output_shapes
:	 :!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::%	!

_output_shapes
:	 : 


_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

: : 

_output_shapes
: :%!

_output_shapes
:	 :!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::% !

_output_shapes
:	 : !

_output_shapes
: :$" 

_output_shapes

: : #

_output_shapes
::$$ 

_output_shapes

: : %

_output_shapes
: :%&!

_output_shapes
:	 :!'

_output_shapes	
::&("
 
_output_shapes
:
:!)

_output_shapes	
::&*"
 
_output_shapes
:
:!+

_output_shapes	
::%,!

_output_shapes
:	 : -

_output_shapes
: :$. 

_output_shapes

: : /

_output_shapes
::0

_output_shapes
: 
ïG
Å
 __inference__wrapped_model_11524
input_1L
:sequential_1st_hidden_layer_matmul_readvariableop_resource: I
;sequential_1st_hidden_layer_biasadd_readvariableop_resource: M
:sequential_2nd_hidden_layer_matmul_readvariableop_resource:	 J
;sequential_2nd_hidden_layer_biasadd_readvariableop_resource:	N
:sequential_3rd_hidden_layer_matmul_readvariableop_resource:
J
;sequential_3rd_hidden_layer_biasadd_readvariableop_resource:	N
:sequential_4th_hidden_layer_matmul_readvariableop_resource:
J
;sequential_4th_hidden_layer_biasadd_readvariableop_resource:	M
:sequential_5th_hidden_layer_matmul_readvariableop_resource:	 I
;sequential_5th_hidden_layer_biasadd_readvariableop_resource: H
6sequential_output_layer_matmul_readvariableop_resource: E
7sequential_output_layer_biasadd_readvariableop_resource:
identity¢2sequential/1st_Hidden_Layer/BiasAdd/ReadVariableOp¢1sequential/1st_Hidden_Layer/MatMul/ReadVariableOp¢2sequential/2nd_Hidden_Layer/BiasAdd/ReadVariableOp¢1sequential/2nd_Hidden_Layer/MatMul/ReadVariableOp¢2sequential/3rd_Hidden_Layer/BiasAdd/ReadVariableOp¢1sequential/3rd_Hidden_Layer/MatMul/ReadVariableOp¢2sequential/4th_Hidden_Layer/BiasAdd/ReadVariableOp¢1sequential/4th_Hidden_Layer/MatMul/ReadVariableOp¢2sequential/5th_Hidden_Layer/BiasAdd/ReadVariableOp¢1sequential/5th_Hidden_Layer/MatMul/ReadVariableOp¢.sequential/Output_Layer/BiasAdd/ReadVariableOp¢-sequential/Output_Layer/MatMul/ReadVariableOp¬
1sequential/1st_Hidden_Layer/MatMul/ReadVariableOpReadVariableOp:sequential_1st_hidden_layer_matmul_readvariableop_resource*
_output_shapes

: *
dtype0¢
"sequential/1st_Hidden_Layer/MatMulMatMulinput_19sequential/1st_Hidden_Layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ª
2sequential/1st_Hidden_Layer/BiasAdd/ReadVariableOpReadVariableOp;sequential_1st_hidden_layer_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ê
#sequential/1st_Hidden_Layer/BiasAddBiasAdd,sequential/1st_Hidden_Layer/MatMul:product:0:sequential/1st_Hidden_Layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 sequential/1st_Hidden_Layer/ReluRelu,sequential/1st_Hidden_Layer/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ­
1sequential/2nd_Hidden_Layer/MatMul/ReadVariableOpReadVariableOp:sequential_2nd_hidden_layer_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype0Ê
"sequential/2nd_Hidden_Layer/MatMulMatMul.sequential/1st_Hidden_Layer/Relu:activations:09sequential/2nd_Hidden_Layer/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ«
2sequential/2nd_Hidden_Layer/BiasAdd/ReadVariableOpReadVariableOp;sequential_2nd_hidden_layer_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ë
#sequential/2nd_Hidden_Layer/BiasAddBiasAdd,sequential/2nd_Hidden_Layer/MatMul:product:0:sequential/2nd_Hidden_Layer/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 sequential/2nd_Hidden_Layer/ReluRelu,sequential/2nd_Hidden_Layer/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
1sequential/3rd_Hidden_Layer/MatMul/ReadVariableOpReadVariableOp:sequential_3rd_hidden_layer_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0Ê
"sequential/3rd_Hidden_Layer/MatMulMatMul.sequential/2nd_Hidden_Layer/Relu:activations:09sequential/3rd_Hidden_Layer/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ«
2sequential/3rd_Hidden_Layer/BiasAdd/ReadVariableOpReadVariableOp;sequential_3rd_hidden_layer_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ë
#sequential/3rd_Hidden_Layer/BiasAddBiasAdd,sequential/3rd_Hidden_Layer/MatMul:product:0:sequential/3rd_Hidden_Layer/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 sequential/3rd_Hidden_Layer/ReluRelu,sequential/3rd_Hidden_Layer/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
1sequential/4th_Hidden_Layer/MatMul/ReadVariableOpReadVariableOp:sequential_4th_hidden_layer_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0Ê
"sequential/4th_Hidden_Layer/MatMulMatMul.sequential/3rd_Hidden_Layer/Relu:activations:09sequential/4th_Hidden_Layer/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ«
2sequential/4th_Hidden_Layer/BiasAdd/ReadVariableOpReadVariableOp;sequential_4th_hidden_layer_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ë
#sequential/4th_Hidden_Layer/BiasAddBiasAdd,sequential/4th_Hidden_Layer/MatMul:product:0:sequential/4th_Hidden_Layer/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 sequential/4th_Hidden_Layer/ReluRelu,sequential/4th_Hidden_Layer/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ­
1sequential/5th_Hidden_Layer/MatMul/ReadVariableOpReadVariableOp:sequential_5th_hidden_layer_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype0É
"sequential/5th_Hidden_Layer/MatMulMatMul.sequential/4th_Hidden_Layer/Relu:activations:09sequential/5th_Hidden_Layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ª
2sequential/5th_Hidden_Layer/BiasAdd/ReadVariableOpReadVariableOp;sequential_5th_hidden_layer_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ê
#sequential/5th_Hidden_Layer/BiasAddBiasAdd,sequential/5th_Hidden_Layer/MatMul:product:0:sequential/5th_Hidden_Layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 sequential/5th_Hidden_Layer/ReluRelu,sequential/5th_Hidden_Layer/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¤
-sequential/Output_Layer/MatMul/ReadVariableOpReadVariableOp6sequential_output_layer_matmul_readvariableop_resource*
_output_shapes

: *
dtype0Á
sequential/Output_Layer/MatMulMatMul.sequential/5th_Hidden_Layer/Relu:activations:05sequential/Output_Layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
.sequential/Output_Layer/BiasAdd/ReadVariableOpReadVariableOp7sequential_output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¾
sequential/Output_Layer/BiasAddBiasAdd(sequential/Output_Layer/MatMul:product:06sequential/Output_Layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
sequential/Output_Layer/SigmoidSigmoid(sequential/Output_Layer/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
IdentityIdentity#sequential/Output_Layer/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
NoOpNoOp3^sequential/1st_Hidden_Layer/BiasAdd/ReadVariableOp2^sequential/1st_Hidden_Layer/MatMul/ReadVariableOp3^sequential/2nd_Hidden_Layer/BiasAdd/ReadVariableOp2^sequential/2nd_Hidden_Layer/MatMul/ReadVariableOp3^sequential/3rd_Hidden_Layer/BiasAdd/ReadVariableOp2^sequential/3rd_Hidden_Layer/MatMul/ReadVariableOp3^sequential/4th_Hidden_Layer/BiasAdd/ReadVariableOp2^sequential/4th_Hidden_Layer/MatMul/ReadVariableOp3^sequential/5th_Hidden_Layer/BiasAdd/ReadVariableOp2^sequential/5th_Hidden_Layer/MatMul/ReadVariableOp/^sequential/Output_Layer/BiasAdd/ReadVariableOp.^sequential/Output_Layer/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 2h
2sequential/1st_Hidden_Layer/BiasAdd/ReadVariableOp2sequential/1st_Hidden_Layer/BiasAdd/ReadVariableOp2f
1sequential/1st_Hidden_Layer/MatMul/ReadVariableOp1sequential/1st_Hidden_Layer/MatMul/ReadVariableOp2h
2sequential/2nd_Hidden_Layer/BiasAdd/ReadVariableOp2sequential/2nd_Hidden_Layer/BiasAdd/ReadVariableOp2f
1sequential/2nd_Hidden_Layer/MatMul/ReadVariableOp1sequential/2nd_Hidden_Layer/MatMul/ReadVariableOp2h
2sequential/3rd_Hidden_Layer/BiasAdd/ReadVariableOp2sequential/3rd_Hidden_Layer/BiasAdd/ReadVariableOp2f
1sequential/3rd_Hidden_Layer/MatMul/ReadVariableOp1sequential/3rd_Hidden_Layer/MatMul/ReadVariableOp2h
2sequential/4th_Hidden_Layer/BiasAdd/ReadVariableOp2sequential/4th_Hidden_Layer/BiasAdd/ReadVariableOp2f
1sequential/4th_Hidden_Layer/MatMul/ReadVariableOp1sequential/4th_Hidden_Layer/MatMul/ReadVariableOp2h
2sequential/5th_Hidden_Layer/BiasAdd/ReadVariableOp2sequential/5th_Hidden_Layer/BiasAdd/ReadVariableOp2f
1sequential/5th_Hidden_Layer/MatMul/ReadVariableOp1sequential/5th_Hidden_Layer/MatMul/ReadVariableOp2`
.sequential/Output_Layer/BiasAdd/ReadVariableOp.sequential/Output_Layer/BiasAdd/ReadVariableOp2^
-sequential/Output_Layer/MatMul/ReadVariableOp-sequential/Output_Layer/MatMul/ReadVariableOp:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
¢

ü
K__inference_1st_Hidden_Layer_layer_call_and_return_conditional_losses_11542

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


ø
G__inference_Output_Layer_layer_call_and_return_conditional_losses_11627

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
$
¹
E__inference_sequential_layer_call_and_return_conditional_losses_11786

inputs'
st_hidden_layer_11755: #
st_hidden_layer_11757: (
nd_hidden_layer_11760:	 $
nd_hidden_layer_11762:	)
rd_hidden_layer_11765:
$
rd_hidden_layer_11767:	)
th_hidden_layer_11770:
$
th_hidden_layer_11772:	(
th_hidden_layer_11775:	 #
th_hidden_layer_11777: $
output_layer_11780:  
output_layer_11782:
identity¢(1st_Hidden_Layer/StatefulPartitionedCall¢(2nd_Hidden_Layer/StatefulPartitionedCall¢(3rd_Hidden_Layer/StatefulPartitionedCall¢(4th_Hidden_Layer/StatefulPartitionedCall¢(5th_Hidden_Layer/StatefulPartitionedCall¢$Output_Layer/StatefulPartitionedCall
(1st_Hidden_Layer/StatefulPartitionedCallStatefulPartitionedCallinputsst_hidden_layer_11755st_hidden_layer_11757*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_1st_Hidden_Layer_layer_call_and_return_conditional_losses_11542º
(2nd_Hidden_Layer/StatefulPartitionedCallStatefulPartitionedCall11st_Hidden_Layer/StatefulPartitionedCall:output:0nd_hidden_layer_11760nd_hidden_layer_11762*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_2nd_Hidden_Layer_layer_call_and_return_conditional_losses_11559º
(3rd_Hidden_Layer/StatefulPartitionedCallStatefulPartitionedCall12nd_Hidden_Layer/StatefulPartitionedCall:output:0rd_hidden_layer_11765rd_hidden_layer_11767*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_3rd_Hidden_Layer_layer_call_and_return_conditional_losses_11576º
(4th_Hidden_Layer/StatefulPartitionedCallStatefulPartitionedCall13rd_Hidden_Layer/StatefulPartitionedCall:output:0th_hidden_layer_11770th_hidden_layer_11772*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_4th_Hidden_Layer_layer_call_and_return_conditional_losses_11593¹
(5th_Hidden_Layer/StatefulPartitionedCallStatefulPartitionedCall14th_Hidden_Layer/StatefulPartitionedCall:output:0th_hidden_layer_11775th_hidden_layer_11777*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_5th_Hidden_Layer_layer_call_and_return_conditional_losses_11610«
$Output_Layer/StatefulPartitionedCallStatefulPartitionedCall15th_Hidden_Layer/StatefulPartitionedCall:output:0output_layer_11780output_layer_11782*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_Output_Layer_layer_call_and_return_conditional_losses_11627|
IdentityIdentity-Output_Layer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
NoOpNoOp)^1st_Hidden_Layer/StatefulPartitionedCall)^2nd_Hidden_Layer/StatefulPartitionedCall)^3rd_Hidden_Layer/StatefulPartitionedCall)^4th_Hidden_Layer/StatefulPartitionedCall)^5th_Hidden_Layer/StatefulPartitionedCall%^Output_Layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 2T
(1st_Hidden_Layer/StatefulPartitionedCall(1st_Hidden_Layer/StatefulPartitionedCall2T
(2nd_Hidden_Layer/StatefulPartitionedCall(2nd_Hidden_Layer/StatefulPartitionedCall2T
(3rd_Hidden_Layer/StatefulPartitionedCall(3rd_Hidden_Layer/StatefulPartitionedCall2T
(4th_Hidden_Layer/StatefulPartitionedCall(4th_Hidden_Layer/StatefulPartitionedCall2T
(5th_Hidden_Layer/StatefulPartitionedCall(5th_Hidden_Layer/StatefulPartitionedCall2L
$Output_Layer/StatefulPartitionedCall$Output_Layer/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ù

°
*__inference_sequential_layer_call_fn_11842
input_1
unknown: 
	unknown_0: 
	unknown_1:	 
	unknown_2:	
	unknown_3:

	unknown_4:	
	unknown_5:

	unknown_6:	
	unknown_7:	 
	unknown_8: 
	unknown_9: 

unknown_10:
identity¢StatefulPartitionedCallá
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_11786o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
ª

þ
K__inference_2nd_Hidden_Layer_layer_call_and_return_conditional_losses_12137

inputs1
matmul_readvariableop_resource:	 .
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	 *
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ö

¯
*__inference_sequential_layer_call_fn_11974

inputs
unknown: 
	unknown_0: 
	unknown_1:	 
	unknown_2:	
	unknown_3:

	unknown_4:	
	unknown_5:

	unknown_6:	
	unknown_7:	 
	unknown_8: 
	unknown_9: 

unknown_10:
identity¢StatefulPartitionedCallà
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_11786o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ó<
Û

E__inference_sequential_layer_call_and_return_conditional_losses_12066

inputs@
.st_hidden_layer_matmul_readvariableop_resource: =
/st_hidden_layer_biasadd_readvariableop_resource: A
.nd_hidden_layer_matmul_readvariableop_resource:	 >
/nd_hidden_layer_biasadd_readvariableop_resource:	B
.rd_hidden_layer_matmul_readvariableop_resource:
>
/rd_hidden_layer_biasadd_readvariableop_resource:	B
.th_hidden_layer_matmul_readvariableop_resource:
>
/th_hidden_layer_biasadd_readvariableop_resource:	C
0th_hidden_layer_matmul_readvariableop_resource_0:	 ?
1th_hidden_layer_biasadd_readvariableop_resource_0: =
+output_layer_matmul_readvariableop_resource: :
,output_layer_biasadd_readvariableop_resource:
identity¢'1st_Hidden_Layer/BiasAdd/ReadVariableOp¢&1st_Hidden_Layer/MatMul/ReadVariableOp¢'2nd_Hidden_Layer/BiasAdd/ReadVariableOp¢&2nd_Hidden_Layer/MatMul/ReadVariableOp¢'3rd_Hidden_Layer/BiasAdd/ReadVariableOp¢&3rd_Hidden_Layer/MatMul/ReadVariableOp¢'4th_Hidden_Layer/BiasAdd/ReadVariableOp¢&4th_Hidden_Layer/MatMul/ReadVariableOp¢'5th_Hidden_Layer/BiasAdd/ReadVariableOp¢&5th_Hidden_Layer/MatMul/ReadVariableOp¢#Output_Layer/BiasAdd/ReadVariableOp¢"Output_Layer/MatMul/ReadVariableOp
&1st_Hidden_Layer/MatMul/ReadVariableOpReadVariableOp.st_hidden_layer_matmul_readvariableop_resource*
_output_shapes

: *
dtype0
1st_Hidden_Layer/MatMulMatMulinputs.1st_Hidden_Layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
'1st_Hidden_Layer/BiasAdd/ReadVariableOpReadVariableOp/st_hidden_layer_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0©
1st_Hidden_Layer/BiasAddBiasAdd!1st_Hidden_Layer/MatMul:product:0/1st_Hidden_Layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
1st_Hidden_Layer/ReluRelu!1st_Hidden_Layer/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
&2nd_Hidden_Layer/MatMul/ReadVariableOpReadVariableOp.nd_hidden_layer_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype0©
2nd_Hidden_Layer/MatMulMatMul#1st_Hidden_Layer/Relu:activations:0.2nd_Hidden_Layer/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'2nd_Hidden_Layer/BiasAdd/ReadVariableOpReadVariableOp/nd_hidden_layer_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ª
2nd_Hidden_Layer/BiasAddBiasAdd!2nd_Hidden_Layer/MatMul:product:0/2nd_Hidden_Layer/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
2nd_Hidden_Layer/ReluRelu!2nd_Hidden_Layer/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&3rd_Hidden_Layer/MatMul/ReadVariableOpReadVariableOp.rd_hidden_layer_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0©
3rd_Hidden_Layer/MatMulMatMul#2nd_Hidden_Layer/Relu:activations:0.3rd_Hidden_Layer/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'3rd_Hidden_Layer/BiasAdd/ReadVariableOpReadVariableOp/rd_hidden_layer_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ª
3rd_Hidden_Layer/BiasAddBiasAdd!3rd_Hidden_Layer/MatMul:product:0/3rd_Hidden_Layer/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
3rd_Hidden_Layer/ReluRelu!3rd_Hidden_Layer/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&4th_Hidden_Layer/MatMul/ReadVariableOpReadVariableOp.th_hidden_layer_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0©
4th_Hidden_Layer/MatMulMatMul#3rd_Hidden_Layer/Relu:activations:0.4th_Hidden_Layer/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'4th_Hidden_Layer/BiasAdd/ReadVariableOpReadVariableOp/th_hidden_layer_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ª
4th_Hidden_Layer/BiasAddBiasAdd!4th_Hidden_Layer/MatMul:product:0/4th_Hidden_Layer/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
4th_Hidden_Layer/ReluRelu!4th_Hidden_Layer/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&5th_Hidden_Layer/MatMul/ReadVariableOpReadVariableOp0th_hidden_layer_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype0¨
5th_Hidden_Layer/MatMulMatMul#4th_Hidden_Layer/Relu:activations:0.5th_Hidden_Layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
'5th_Hidden_Layer/BiasAdd/ReadVariableOpReadVariableOp1th_hidden_layer_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0©
5th_Hidden_Layer/BiasAddBiasAdd!5th_Hidden_Layer/MatMul:product:0/5th_Hidden_Layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
5th_Hidden_Layer/ReluRelu!5th_Hidden_Layer/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"Output_Layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

: *
dtype0 
Output_Layer/MatMulMatMul#5th_Hidden_Layer/Relu:activations:0*Output_Layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#Output_Layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
Output_Layer/BiasAddBiasAddOutput_Layer/MatMul:product:0+Output_Layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
Output_Layer/SigmoidSigmoidOutput_Layer/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
IdentityIdentityOutput_Layer/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
NoOpNoOp(^1st_Hidden_Layer/BiasAdd/ReadVariableOp'^1st_Hidden_Layer/MatMul/ReadVariableOp(^2nd_Hidden_Layer/BiasAdd/ReadVariableOp'^2nd_Hidden_Layer/MatMul/ReadVariableOp(^3rd_Hidden_Layer/BiasAdd/ReadVariableOp'^3rd_Hidden_Layer/MatMul/ReadVariableOp(^4th_Hidden_Layer/BiasAdd/ReadVariableOp'^4th_Hidden_Layer/MatMul/ReadVariableOp(^5th_Hidden_Layer/BiasAdd/ReadVariableOp'^5th_Hidden_Layer/MatMul/ReadVariableOp$^Output_Layer/BiasAdd/ReadVariableOp#^Output_Layer/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 2R
'1st_Hidden_Layer/BiasAdd/ReadVariableOp'1st_Hidden_Layer/BiasAdd/ReadVariableOp2P
&1st_Hidden_Layer/MatMul/ReadVariableOp&1st_Hidden_Layer/MatMul/ReadVariableOp2R
'2nd_Hidden_Layer/BiasAdd/ReadVariableOp'2nd_Hidden_Layer/BiasAdd/ReadVariableOp2P
&2nd_Hidden_Layer/MatMul/ReadVariableOp&2nd_Hidden_Layer/MatMul/ReadVariableOp2R
'3rd_Hidden_Layer/BiasAdd/ReadVariableOp'3rd_Hidden_Layer/BiasAdd/ReadVariableOp2P
&3rd_Hidden_Layer/MatMul/ReadVariableOp&3rd_Hidden_Layer/MatMul/ReadVariableOp2R
'4th_Hidden_Layer/BiasAdd/ReadVariableOp'4th_Hidden_Layer/BiasAdd/ReadVariableOp2P
&4th_Hidden_Layer/MatMul/ReadVariableOp&4th_Hidden_Layer/MatMul/ReadVariableOp2R
'5th_Hidden_Layer/BiasAdd/ReadVariableOp'5th_Hidden_Layer/BiasAdd/ReadVariableOp2P
&5th_Hidden_Layer/MatMul/ReadVariableOp&5th_Hidden_Layer/MatMul/ReadVariableOp2J
#Output_Layer/BiasAdd/ReadVariableOp#Output_Layer/BiasAdd/ReadVariableOp2H
"Output_Layer/MatMul/ReadVariableOp"Output_Layer/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ó

0__inference_1st_Hidden_Layer_layer_call_fn_12106

inputs
unknown: 
	unknown_0: 
identity¢StatefulPartitionedCallã
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_1st_Hidden_Layer_layer_call_and_return_conditional_losses_11542o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ª

þ
K__inference_2nd_Hidden_Layer_layer_call_and_return_conditional_losses_11559

inputs1
matmul_readvariableop_resource:	 .
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	 *
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
®

ÿ
K__inference_3rd_Hidden_Layer_layer_call_and_return_conditional_losses_12157

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
®

ÿ
K__inference_4th_Hidden_Layer_layer_call_and_return_conditional_losses_12177

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ú
 
0__inference_4th_Hidden_Layer_layer_call_fn_12166

inputs
unknown:

	unknown_0:	
identity¢StatefulPartitionedCallä
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_4th_Hidden_Layer_layer_call_and_return_conditional_losses_11593p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
$
º
E__inference_sequential_layer_call_and_return_conditional_losses_11876
input_1'
st_hidden_layer_11845: #
st_hidden_layer_11847: (
nd_hidden_layer_11850:	 $
nd_hidden_layer_11852:	)
rd_hidden_layer_11855:
$
rd_hidden_layer_11857:	)
th_hidden_layer_11860:
$
th_hidden_layer_11862:	(
th_hidden_layer_11865:	 #
th_hidden_layer_11867: $
output_layer_11870:  
output_layer_11872:
identity¢(1st_Hidden_Layer/StatefulPartitionedCall¢(2nd_Hidden_Layer/StatefulPartitionedCall¢(3rd_Hidden_Layer/StatefulPartitionedCall¢(4th_Hidden_Layer/StatefulPartitionedCall¢(5th_Hidden_Layer/StatefulPartitionedCall¢$Output_Layer/StatefulPartitionedCall
(1st_Hidden_Layer/StatefulPartitionedCallStatefulPartitionedCallinput_1st_hidden_layer_11845st_hidden_layer_11847*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_1st_Hidden_Layer_layer_call_and_return_conditional_losses_11542º
(2nd_Hidden_Layer/StatefulPartitionedCallStatefulPartitionedCall11st_Hidden_Layer/StatefulPartitionedCall:output:0nd_hidden_layer_11850nd_hidden_layer_11852*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_2nd_Hidden_Layer_layer_call_and_return_conditional_losses_11559º
(3rd_Hidden_Layer/StatefulPartitionedCallStatefulPartitionedCall12nd_Hidden_Layer/StatefulPartitionedCall:output:0rd_hidden_layer_11855rd_hidden_layer_11857*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_3rd_Hidden_Layer_layer_call_and_return_conditional_losses_11576º
(4th_Hidden_Layer/StatefulPartitionedCallStatefulPartitionedCall13rd_Hidden_Layer/StatefulPartitionedCall:output:0th_hidden_layer_11860th_hidden_layer_11862*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_4th_Hidden_Layer_layer_call_and_return_conditional_losses_11593¹
(5th_Hidden_Layer/StatefulPartitionedCallStatefulPartitionedCall14th_Hidden_Layer/StatefulPartitionedCall:output:0th_hidden_layer_11865th_hidden_layer_11867*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_5th_Hidden_Layer_layer_call_and_return_conditional_losses_11610«
$Output_Layer/StatefulPartitionedCallStatefulPartitionedCall15th_Hidden_Layer/StatefulPartitionedCall:output:0output_layer_11870output_layer_11872*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_Output_Layer_layer_call_and_return_conditional_losses_11627|
IdentityIdentity-Output_Layer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
NoOpNoOp)^1st_Hidden_Layer/StatefulPartitionedCall)^2nd_Hidden_Layer/StatefulPartitionedCall)^3rd_Hidden_Layer/StatefulPartitionedCall)^4th_Hidden_Layer/StatefulPartitionedCall)^5th_Hidden_Layer/StatefulPartitionedCall%^Output_Layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 2T
(1st_Hidden_Layer/StatefulPartitionedCall(1st_Hidden_Layer/StatefulPartitionedCall2T
(2nd_Hidden_Layer/StatefulPartitionedCall(2nd_Hidden_Layer/StatefulPartitionedCall2T
(3rd_Hidden_Layer/StatefulPartitionedCall(3rd_Hidden_Layer/StatefulPartitionedCall2T
(4th_Hidden_Layer/StatefulPartitionedCall(4th_Hidden_Layer/StatefulPartitionedCall2T
(5th_Hidden_Layer/StatefulPartitionedCall(5th_Hidden_Layer/StatefulPartitionedCall2L
$Output_Layer/StatefulPartitionedCall$Output_Layer/StatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1"ÛL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*¯
serving_default
;
input_10
serving_default_input_1:0ÿÿÿÿÿÿÿÿÿ@
Output_Layer0
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:ð
Ð
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
layer_with_weights-5
layer-5
	optimizer
	variables
	trainable_variables

regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_sequential
»

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
»

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
»

 kernel
!bias
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses"
_tf_keras_layer
»

(kernel
)bias
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses"
_tf_keras_layer
»

0kernel
1bias
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses"
_tf_keras_layer
»

8kernel
9bias
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses"
_tf_keras_layer
¹
@iter

Abeta_1

Bbeta_2
	Cdecay
Dlearning_ratemvmwmxmy mz!m{(m|)m}0m~1m8m9mvvvv v!v(v)v0v1v8v9v"
	optimizer
v
0
1
2
3
 4
!5
(6
)7
08
19
810
911"
trackable_list_wrapper
v
0
1
2
3
 4
!5
(6
)7
08
19
810
911"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
Enon_trainable_variables

Flayers
Gmetrics
Hlayer_regularization_losses
Ilayer_metrics
	variables
	trainable_variables

regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ö2ó
*__inference_sequential_layer_call_fn_11661
*__inference_sequential_layer_call_fn_11945
*__inference_sequential_layer_call_fn_11974
*__inference_sequential_layer_call_fn_11842À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
â2ß
E__inference_sequential_layer_call_and_return_conditional_losses_12020
E__inference_sequential_layer_call_and_return_conditional_losses_12066
E__inference_sequential_layer_call_and_return_conditional_losses_11876
E__inference_sequential_layer_call_and_return_conditional_losses_11910À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ËBÈ
 __inference__wrapped_model_11524input_1"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
,
Jserving_default"
signature_map
):' 21st_Hidden_Layer/kernel
#:! 21st_Hidden_Layer/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Knon_trainable_variables

Llayers
Mmetrics
Nlayer_regularization_losses
Olayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ú2×
0__inference_1st_Hidden_Layer_layer_call_fn_12106¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
õ2ò
K__inference_1st_Hidden_Layer_layer_call_and_return_conditional_losses_12117¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
*:(	 22nd_Hidden_Layer/kernel
$:"22nd_Hidden_Layer/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Pnon_trainable_variables

Qlayers
Rmetrics
Slayer_regularization_losses
Tlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ú2×
0__inference_2nd_Hidden_Layer_layer_call_fn_12126¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
õ2ò
K__inference_2nd_Hidden_Layer_layer_call_and_return_conditional_losses_12137¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
+:)
23rd_Hidden_Layer/kernel
$:"23rd_Hidden_Layer/bias
.
 0
!1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Unon_trainable_variables

Vlayers
Wmetrics
Xlayer_regularization_losses
Ylayer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses"
_generic_user_object
Ú2×
0__inference_3rd_Hidden_Layer_layer_call_fn_12146¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
õ2ò
K__inference_3rd_Hidden_Layer_layer_call_and_return_conditional_losses_12157¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
+:)
24th_Hidden_Layer/kernel
$:"24th_Hidden_Layer/bias
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Znon_trainable_variables

[layers
\metrics
]layer_regularization_losses
^layer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses"
_generic_user_object
Ú2×
0__inference_4th_Hidden_Layer_layer_call_fn_12166¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
õ2ò
K__inference_4th_Hidden_Layer_layer_call_and_return_conditional_losses_12177¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
*:(	 25th_Hidden_Layer/kernel
#:! 25th_Hidden_Layer/bias
.
00
11"
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
 "
trackable_list_wrapper
­
_non_trainable_variables

`layers
ametrics
blayer_regularization_losses
clayer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses"
_generic_user_object
Ú2×
0__inference_5th_Hidden_Layer_layer_call_fn_12186¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
õ2ò
K__inference_5th_Hidden_Layer_layer_call_and_return_conditional_losses_12197¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
%:# 2Output_Layer/kernel
:2Output_Layer/bias
.
80
91"
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
 "
trackable_list_wrapper
­
dnon_trainable_variables

elayers
fmetrics
glayer_regularization_losses
hlayer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses"
_generic_user_object
Ö2Ó
,__inference_Output_Layer_layer_call_fn_12206¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_Output_Layer_layer_call_and_return_conditional_losses_12217¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
.
i0
j1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÊBÇ
#__inference_signature_wrapper_12097input_1"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
N
	ktotal
	lcount
m	variables
n	keras_api"
_tf_keras_metric
 
o
init_shape
ptrue_positives
qfalse_positives
rfalse_negatives
sweights_intermediate
t	variables
u	keras_api"
_tf_keras_metric
:  (2total
:  (2count
.
k0
l1"
trackable_list_wrapper
-
m	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_positives
: (2false_negatives
$:" (2weights_intermediate
<
p0
q1
r2
s3"
trackable_list_wrapper
-
t	variables"
_generic_user_object
.:, 2Adam/1st_Hidden_Layer/kernel/m
(:& 2Adam/1st_Hidden_Layer/bias/m
/:-	 2Adam/2nd_Hidden_Layer/kernel/m
):'2Adam/2nd_Hidden_Layer/bias/m
0:.
2Adam/3rd_Hidden_Layer/kernel/m
):'2Adam/3rd_Hidden_Layer/bias/m
0:.
2Adam/4th_Hidden_Layer/kernel/m
):'2Adam/4th_Hidden_Layer/bias/m
/:-	 2Adam/5th_Hidden_Layer/kernel/m
(:& 2Adam/5th_Hidden_Layer/bias/m
*:( 2Adam/Output_Layer/kernel/m
$:"2Adam/Output_Layer/bias/m
.:, 2Adam/1st_Hidden_Layer/kernel/v
(:& 2Adam/1st_Hidden_Layer/bias/v
/:-	 2Adam/2nd_Hidden_Layer/kernel/v
):'2Adam/2nd_Hidden_Layer/bias/v
0:.
2Adam/3rd_Hidden_Layer/kernel/v
):'2Adam/3rd_Hidden_Layer/bias/v
0:.
2Adam/4th_Hidden_Layer/kernel/v
):'2Adam/4th_Hidden_Layer/bias/v
/:-	 2Adam/5th_Hidden_Layer/kernel/v
(:& 2Adam/5th_Hidden_Layer/bias/v
*:( 2Adam/Output_Layer/kernel/v
$:"2Adam/Output_Layer/bias/v«
K__inference_1st_Hidden_Layer_layer_call_and_return_conditional_losses_12117\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 
0__inference_1st_Hidden_Layer_layer_call_fn_12106O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ ¬
K__inference_2nd_Hidden_Layer_layer_call_and_return_conditional_losses_12137]/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
0__inference_2nd_Hidden_Layer_layer_call_fn_12126P/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿ­
K__inference_3rd_Hidden_Layer_layer_call_and_return_conditional_losses_12157^ !0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
0__inference_3rd_Hidden_Layer_layer_call_fn_12146Q !0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ­
K__inference_4th_Hidden_Layer_layer_call_and_return_conditional_losses_12177^()0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
0__inference_4th_Hidden_Layer_layer_call_fn_12166Q()0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¬
K__inference_5th_Hidden_Layer_layer_call_and_return_conditional_losses_12197]010¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 
0__inference_5th_Hidden_Layer_layer_call_fn_12186P010¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ §
G__inference_Output_Layer_layer_call_and_return_conditional_losses_12217\89/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_Output_Layer_layer_call_fn_12206O89/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿ¡
 __inference__wrapped_model_11524} !()01890¢-
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ
ª ";ª8
6
Output_Layer&#
Output_Layerÿÿÿÿÿÿÿÿÿ¸
E__inference_sequential_layer_call_and_return_conditional_losses_11876o !()01898¢5
.¢+
!
input_1ÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¸
E__inference_sequential_layer_call_and_return_conditional_losses_11910o !()01898¢5
.¢+
!
input_1ÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ·
E__inference_sequential_layer_call_and_return_conditional_losses_12020n !()01897¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ·
E__inference_sequential_layer_call_and_return_conditional_losses_12066n !()01897¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
*__inference_sequential_layer_call_fn_11661b !()01898¢5
.¢+
!
input_1ÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
*__inference_sequential_layer_call_fn_11842b !()01898¢5
.¢+
!
input_1ÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
*__inference_sequential_layer_call_fn_11945a !()01897¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
*__inference_sequential_layer_call_fn_11974a !()01897¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ°
#__inference_signature_wrapper_12097 !()0189;¢8
¢ 
1ª.
,
input_1!
input_1ÿÿÿÿÿÿÿÿÿ";ª8
6
Output_Layer&#
Output_Layerÿÿÿÿÿÿÿÿÿ