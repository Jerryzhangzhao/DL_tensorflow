E
Ã
:
Add
x"T
y"T
z"T"
Ttype:
2	
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
8
Const
output"dtype"
valuetensor"
dtypetype
ì
Conv2D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
p
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
	2
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
=
Mul
x"T
y"T
z"T"
Ttype:
2	
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
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
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
H
ShardedFilename
basename	
shard

num_shards
filename
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

TruncatedNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring "MODEL_TRAINING*1.12.02v1.12.0-0-ga6d8ffae09É3
X
x_inputPlaceholder*
dtype0*
_output_shapes

:*
shape
:
h
x_reshape/shapeConst*%
valueB"ÿÿÿÿ         *
dtype0*
_output_shapes
:
m
	x_reshapeReshapex_inputx_reshape/shape*&
_output_shapes
:*
T0*
Tshape0
o
truncated_normal/shapeConst*
_output_shapes
:*%
valueB"            *
dtype0
Z
truncated_normal/meanConst*
_output_shapes
: *
valueB
 *    *
dtype0
\
truncated_normal/stddevConst*
valueB
 *ÍÌÌ=*
dtype0*
_output_shapes
: 
¢
 truncated_normal/TruncatedNormalTruncatedNormaltruncated_normal/shape*
T0*
dtype0*&
_output_shapes
:*
seed2 *

seed 

truncated_normal/mulMul truncated_normal/TruncatedNormaltruncated_normal/stddev*
T0*&
_output_shapes
:
u
truncated_normalAddtruncated_normal/multruncated_normal/mean*
T0*&
_output_shapes
:

w
VariableV2*
shape:*
shared_name *
dtype0*&
_output_shapes
:*
	container 

w/AssignAssignwtruncated_normal*
_class

loc:@w*
validate_shape(*&
_output_shapes
:*
use_locking(*
T0
\
w/readIdentityw*
T0*
_class

loc:@w*&
_output_shapes
:
À
convConv2D	x_reshapew/read*
paddingSAME*&
_output_shapes
:*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
i
truncated_normal_1/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
\
truncated_normal_1/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
^
truncated_normal_1/stddevConst*
_output_shapes
: *
valueB
 *ÍÌÌ=*
dtype0

"truncated_normal_1/TruncatedNormalTruncatedNormaltruncated_normal_1/shape*
T0*
dtype0*
_output_shapes

:*
seed2 *

seed 

truncated_normal_1/mulMul"truncated_normal_1/TruncatedNormaltruncated_normal_1/stddev*
T0*
_output_shapes

:
s
truncated_normal_1Addtruncated_normal_1/multruncated_normal_1/mean*
T0*
_output_shapes

:
v
w1
VariableV2*
shape
:*
shared_name *
dtype0*
_output_shapes

:*
	container 

	w1/AssignAssignw1truncated_normal_1*
use_locking(*
T0*
_class
	loc:@w1*
validate_shape(*
_output_shapes

:
W
w1/readIdentityw1*
T0*
_class
	loc:@w1*
_output_shapes

:
i
truncated_normal_2/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
\
truncated_normal_2/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
^
truncated_normal_2/stddevConst*
valueB
 *ÍÌÌ=*
dtype0*
_output_shapes
: 

"truncated_normal_2/TruncatedNormalTruncatedNormaltruncated_normal_2/shape*
dtype0*
_output_shapes

:*
seed2 *

seed *
T0

truncated_normal_2/mulMul"truncated_normal_2/TruncatedNormaltruncated_normal_2/stddev*
_output_shapes

:*
T0
s
truncated_normal_2Addtruncated_normal_2/multruncated_normal_2/mean*
_output_shapes

:*
T0
v
w2
VariableV2*
shared_name *
dtype0*
_output_shapes

:*
	container *
shape
:

	w2/AssignAssignw2truncated_normal_2*
_output_shapes

:*
use_locking(*
T0*
_class
	loc:@w2*
validate_shape(
W
w2/readIdentityw2*
T0*
_class
	loc:@w2*
_output_shapes

:
l
yMatMulw1/readw2/read*
_output_shapes

:*
transpose_a( *
transpose_b( *
T0
/
initNoOp	^w/Assign
^w1/Assign
^w2/Assign
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 

save/StringJoin/inputs_1Const*<
value3B1 B+_temp_e9d380ef4ec44b3487ab31b6c08a9eb0/part*
dtype0*
_output_shapes
: 
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
_output_shapes
: *
	separator *
N
Q
save/num_shardsConst*
_output_shapes
: *
value	B :*
dtype0
k
save/ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 

save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
y
save/SaveV2/tensor_namesConst"/device:CPU:0*
valueBBwBw1Bw2*
dtype0*
_output_shapes
:
x
save/SaveV2/shape_and_slicesConst"/device:CPU:0*
valueBB B B *
dtype0*
_output_shapes
:

save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesww1w2"/device:CPU:0*
dtypes
2
 
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2"/device:CPU:0*
T0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: 
¬
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency"/device:CPU:0*
T0*

axis *
N*
_output_shapes
:

save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const"/device:CPU:0*
delete_old_dirs(

save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency"/device:CPU:0*
_output_shapes
: *
T0
|
save/RestoreV2/tensor_namesConst"/device:CPU:0*
valueBBwBw1Bw2*
dtype0*
_output_shapes
:
{
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
valueBB B B *
dtype0
©
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0* 
_output_shapes
:::*
dtypes
2

save/AssignAssignwsave/RestoreV2*
validate_shape(*&
_output_shapes
:*
use_locking(*
T0*
_class

loc:@w

save/Assign_1Assignw1save/RestoreV2:1*
_class
	loc:@w1*
validate_shape(*
_output_shapes

:*
use_locking(*
T0

save/Assign_2Assignw2save/RestoreV2:2*
_output_shapes

:*
use_locking(*
T0*
_class
	loc:@w2*
validate_shape(
H
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_2
-
save/restore_allNoOp^save/restore_shard"<
save/Const:0save/Identity:0save/restore_all (5 @F8"¸
trainable_variables 
/
w:0w/Assignw/read:02truncated_normal:08
4
w1:0	w1/Assign	w1/read:02truncated_normal_1:08
4
w2:0	w2/Assign	w2/read:02truncated_normal_2:08"®
	variables 
/
w:0w/Assignw/read:02truncated_normal:08
4
w1:0	w1/Assign	w1/read:02truncated_normal_1:08
4
w2:0	w2/Assign	w2/read:02truncated_normal_2:08*\
my_signatureL
!
input0
	x_input:0'
output0
conv:0[
Â
:
Add
x"T
y"T
z"T"
Ttype:
2	
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
8
Const
output"dtype"
valuetensor"
dtypetype
ì
Conv2D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
p
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
	2
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
=
Mul
x"T
y"T
z"T"
Ttype:
2	
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
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
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
H
ShardedFilename
basename	
shard

num_shards
filename
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

TruncatedNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring "MODEL_SERVING*1.12.02v1.12.0-0-ga6d8ffae09¬I
X
x_inputPlaceholder*
dtype0*
_output_shapes

:*
shape
:
h
x_reshape/shapeConst*%
valueB"ÿÿÿÿ         *
dtype0*
_output_shapes
:
m
	x_reshapeReshapex_inputx_reshape/shape*
T0*
Tshape0*&
_output_shapes
:
o
truncated_normal/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:
Z
truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
\
truncated_normal/stddevConst*
valueB
 *ÍÌÌ=*
dtype0*
_output_shapes
: 
¢
 truncated_normal/TruncatedNormalTruncatedNormaltruncated_normal/shape*

seed *
T0*
dtype0*&
_output_shapes
:*
seed2 

truncated_normal/mulMul truncated_normal/TruncatedNormaltruncated_normal/stddev*
T0*&
_output_shapes
:
u
truncated_normalAddtruncated_normal/multruncated_normal/mean*
T0*&
_output_shapes
:

w
VariableV2*
shape:*
shared_name *
dtype0*&
_output_shapes
:*
	container 

w/AssignAssignwtruncated_normal*
_class

loc:@w*
validate_shape(*&
_output_shapes
:*
use_locking(*
T0
\
w/readIdentityw*&
_output_shapes
:*
T0*
_class

loc:@w
À
convConv2D	x_reshapew/read*&
_output_shapes
:*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
i
truncated_normal_1/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
\
truncated_normal_1/meanConst*
_output_shapes
: *
valueB
 *    *
dtype0
^
truncated_normal_1/stddevConst*
valueB
 *ÍÌÌ=*
dtype0*
_output_shapes
: 

"truncated_normal_1/TruncatedNormalTruncatedNormaltruncated_normal_1/shape*
T0*
dtype0*
_output_shapes

:*
seed2 *

seed 

truncated_normal_1/mulMul"truncated_normal_1/TruncatedNormaltruncated_normal_1/stddev*
T0*
_output_shapes

:
s
truncated_normal_1Addtruncated_normal_1/multruncated_normal_1/mean*
_output_shapes

:*
T0
v
w1
VariableV2*
shared_name *
dtype0*
_output_shapes

:*
	container *
shape
:

	w1/AssignAssignw1truncated_normal_1*
T0*
_class
	loc:@w1*
validate_shape(*
_output_shapes

:*
use_locking(
W
w1/readIdentityw1*
_output_shapes

:*
T0*
_class
	loc:@w1
i
truncated_normal_2/shapeConst*
_output_shapes
:*
valueB"      *
dtype0
\
truncated_normal_2/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
^
truncated_normal_2/stddevConst*
valueB
 *ÍÌÌ=*
dtype0*
_output_shapes
: 

"truncated_normal_2/TruncatedNormalTruncatedNormaltruncated_normal_2/shape*
dtype0*
_output_shapes

:*
seed2 *

seed *
T0

truncated_normal_2/mulMul"truncated_normal_2/TruncatedNormaltruncated_normal_2/stddev*
T0*
_output_shapes

:
s
truncated_normal_2Addtruncated_normal_2/multruncated_normal_2/mean*
_output_shapes

:*
T0
v
w2
VariableV2*
dtype0*
_output_shapes

:*
	container *
shape
:*
shared_name 

	w2/AssignAssignw2truncated_normal_2*
_class
	loc:@w2*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
W
w2/readIdentityw2*
T0*
_class
	loc:@w2*
_output_shapes

:
l
yMatMulw1/readw2/read*
_output_shapes

:*
transpose_a( *
transpose_b( *
T0
/
initNoOp	^w/Assign
^w1/Assign
^w2/Assign
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 

save/StringJoin/inputs_1Const*<
value3B1 B+_temp_e9d380ef4ec44b3487ab31b6c08a9eb0/part*
dtype0*
_output_shapes
: 
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
Q
save/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
k
save/ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 

save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
y
save/SaveV2/tensor_namesConst"/device:CPU:0*
valueBBwBw1Bw2*
dtype0*
_output_shapes
:
x
save/SaveV2/shape_and_slicesConst"/device:CPU:0*
valueBB B B *
dtype0*
_output_shapes
:

save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesww1w2"/device:CPU:0*
dtypes
2
 
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2"/device:CPU:0*
T0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: 
¬
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency"/device:CPU:0*

axis *
N*
_output_shapes
:*
T0

save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const"/device:CPU:0*
delete_old_dirs(

save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency"/device:CPU:0*
_output_shapes
: *
T0
|
save/RestoreV2/tensor_namesConst"/device:CPU:0*
valueBBwBw1Bw2*
dtype0*
_output_shapes
:
{
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B B *
dtype0*
_output_shapes
:
©
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0* 
_output_shapes
:::*
dtypes
2

save/AssignAssignwsave/RestoreV2*
use_locking(*
T0*
_class

loc:@w*
validate_shape(*&
_output_shapes
:

save/Assign_1Assignw1save/RestoreV2:1*
use_locking(*
T0*
_class
	loc:@w1*
validate_shape(*
_output_shapes

:

save/Assign_2Assignw2save/RestoreV2:2*
T0*
_class
	loc:@w2*
validate_shape(*
_output_shapes

:*
use_locking(
H
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_2
-
save/restore_allNoOp^save/restore_shard
R
save_1/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 

save_1/StringJoin/inputs_1Const*<
value3B1 B+_temp_31dc67dac7ec4acdae6b712ac76885b5/part*
dtype0*
_output_shapes
: 
{
save_1/StringJoin
StringJoinsave_1/Constsave_1/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
S
save_1/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
m
save_1/ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 

save_1/ShardedFilenameShardedFilenamesave_1/StringJoinsave_1/ShardedFilename/shardsave_1/num_shards"/device:CPU:0*
_output_shapes
: 
{
save_1/SaveV2/tensor_namesConst"/device:CPU:0*
valueBBwBw1Bw2*
dtype0*
_output_shapes
:
z
save_1/SaveV2/shape_and_slicesConst"/device:CPU:0*
valueBB B B *
dtype0*
_output_shapes
:

save_1/SaveV2SaveV2save_1/ShardedFilenamesave_1/SaveV2/tensor_namessave_1/SaveV2/shape_and_slicesww1w2"/device:CPU:0*
dtypes
2
¨
save_1/control_dependencyIdentitysave_1/ShardedFilename^save_1/SaveV2"/device:CPU:0*
T0*)
_class
loc:@save_1/ShardedFilename*
_output_shapes
: 
²
-save_1/MergeV2Checkpoints/checkpoint_prefixesPacksave_1/ShardedFilename^save_1/control_dependency"/device:CPU:0*
_output_shapes
:*
T0*

axis *
N

save_1/MergeV2CheckpointsMergeV2Checkpoints-save_1/MergeV2Checkpoints/checkpoint_prefixessave_1/Const"/device:CPU:0*
delete_old_dirs(

save_1/IdentityIdentitysave_1/Const^save_1/MergeV2Checkpoints^save_1/control_dependency"/device:CPU:0*
_output_shapes
: *
T0
~
save_1/RestoreV2/tensor_namesConst"/device:CPU:0*
valueBBwBw1Bw2*
dtype0*
_output_shapes
:
}
!save_1/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B B *
dtype0*
_output_shapes
:
±
save_1/RestoreV2	RestoreV2save_1/Constsave_1/RestoreV2/tensor_names!save_1/RestoreV2/shape_and_slices"/device:CPU:0* 
_output_shapes
:::*
dtypes
2

save_1/AssignAssignwsave_1/RestoreV2*
use_locking(*
T0*
_class

loc:@w*
validate_shape(*&
_output_shapes
:

save_1/Assign_1Assignw1save_1/RestoreV2:1*
use_locking(*
T0*
_class
	loc:@w1*
validate_shape(*
_output_shapes

:

save_1/Assign_2Assignw2save_1/RestoreV2:2*
use_locking(*
T0*
_class
	loc:@w2*
validate_shape(*
_output_shapes

:
P
save_1/restore_shardNoOp^save_1/Assign^save_1/Assign_1^save_1/Assign_2
1
save_1/restore_allNoOp^save_1/restore_shard"B
save_1/Const:0save_1/Identity:0save_1/restore_all (5 @F8"®
	variables 
/
w:0w/Assignw/read:02truncated_normal:08
4
w1:0	w1/Assign	w1/read:02truncated_normal_1:08
4
w2:0	w2/Assign	w2/read:02truncated_normal_2:08"¸
trainable_variables 
/
w:0w/Assignw/read:02truncated_normal:08
4
w1:0	w1/Assign	w1/read:02truncated_normal_1:08
4
w2:0	w2/Assign	w2/read:02truncated_normal_2:08*\
my_signatureL
!
input0
	x_input:0'
output0
conv:0