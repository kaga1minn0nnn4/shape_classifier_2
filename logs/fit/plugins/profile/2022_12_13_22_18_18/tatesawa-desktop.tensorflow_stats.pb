"�l
�DeviceResourceApplyAdam"8training/Adam/Adam/update_dense/kernel/ResourceApplyAdam(1��S��@�@9��S��@�@A��S��@�@I��S��@�@Q�<ѥ�?Y�<ѥ�?�Unknown
�DeviceMatMul":training/Adam/gradients/gradients/dense/MatMul_grad/MatMul(1�x�&�d�@9�x�&�d�@A�x�&�d�@I�x�&�d�@Q��*MԾ?Y��R��?�Unknown
DDeviceIDLE"IDLE1� �rhd�@A� �rhd�@QC�D�eN�?Yw�ʒ��?�Unknown
bDeviceMatMul"dense/MatMul(1��"����@9��"����@A��"����@I��"����@Qh���ĳ?Y��:�B�?�Unknown
�DeviceMatMul"<training/Adam/gradients/gradients/dense/MatMul_grad/MatMul_1(1j�t����@9j�t����@Aj�t����@Ij�t����@QᒒnBĳ?Y@I_:��?�Unknown
�DeviceMaxPoolGrad"Htraining/Adam/gradients/gradients/max_pooling2d/MaxPool_grad/MaxPoolGrad(1��/�Pd@9��/�Pd@A��/�Pd@I��/�Pd@Q��]q@�?YJ��`S��?�Unknown
�DeviceConv2DBackpropFilter"Itraining/Adam/gradients/gradients/conv2d/Conv2D_grad/Conv2DBackpropFilter(1�I+�X@9�I+�X@A�I+�X@I�I+�X@Q��(��t?Y/�"�)�?�Unknown
�DeviceReluGrad";training/Adam/gradients/gradients/conv2d/Relu_grad/ReluGrad(1���MbhW@9���MbhW@A���MbhW@I���MbhW@Q|mԈ�s?Y
ˤ=P�?�Unknown
g	Device_FusedConv2D"conv2d/Relu(1^�I�Q@9^�I�Q@A^�I�Q@I^�I�Q@Q�?�eYm?Y�U]
�m�?�Unknown
�
DeviceBiasAddGrad"Atraining/Adam/gradients/gradients/conv2d/BiasAdd_grad/BiasAddGrad(1�K7�A`G@9�K7�A`G@A�K7�A`G@I�K7�A`G@QR�},��c?Y�Ӊ�-��?�Unknown
lDeviceMaxPool"max_pooling2d/MaxPool(1�~j�tCE@9�~j�tCE@A�~j�tCE@I�~j�tCE@QH�%[��a?Y���M���?�Unknown
�DeviceSoftmaxCrossEntropyWithLogits"3loss/dense_1_loss/softmax_cross_entropy_with_logits(1��Q��;@9��Q��;@A��Q��;@I��Q��;@Q��3n�W?Y�)���?�Unknown
dDeviceMatMul"dense_1/MatMul(1�Zd;�3@9�Zd;�3@A�Zd;�3@I�Zd;�3@Q����P?Y��ڦ�?�Unknown
�Device	Transpose"ntraining/Adam/gradients/gradients/max_pooling2d/MaxPool_grad/MaxPoolGrad-2-TransposeNHWCToNCHW-LayoutOptimizer(1��ʡE1@9��ʡE1@A��ʡE1@I��ʡE1@Q���(�L?Y�	w��?�Unknown
�Device	Transpose"=max_pooling2d/MaxPool-0-0-TransposeNCHWToNHWC-LayoutOptimizer(1�K7�A`-@9�K7�A`-@A�K7�A`-@I�K7�A`-@Q�j�ޝH?Yݟ��*��?�Unknown
vDeviceUnknown"_arg_conv2d_input_0_0/_59:_Recv(1���K�#@9���K�#@A���K�#@I���K�#@Q
���1�@?YW[2�Y��?�Unknown
fDeviceSoftmax"dense_1/Softmax(1Zd;� @9Zd;� @AZd;� @IZd;� @Q8�b�:?Yyu'���?�Unknown
�DeviceResourceApplyAdam"6training/Adam/Adam/update_dense/bias/ResourceApplyAdam(1#��~j�@9#��~j�@A#��~j�@I#��~j�@Q�һ�``1?Y�M�3��?�Unknown
�DeviceMatMul">training/Adam/gradients/gradients/dense_1/MatMul_grad/MatMul_1(1j�t�@9j�t�@Aj�t�@Ij�t�@Q�	�J�0?Y��=���?�Unknown
�DeviceResourceApplyAdam":training/Adam/Adam/update_dense_1/kernel/ResourceApplyAdam(1d;�O�@9d;�O�@Ad;�O�@Id;�O�@Q�~9�9�0?Y�eq���?�Unknown
�DeviceMatMul"<training/Adam/gradients/gradients/dense_1/MatMul_grad/MatMul(1�t��@9�t��@A�t��@I�t��@Q<�V�N�0?Y��B�,��?�Unknown
mDeviceArgMax"metrics/accuracy/ArgMax(1/�$��@9/�$��@A/�$��@I/�$��@QW��$?i0?Y�h'�9��?�Unknown
�DeviceResourceApplyAdam"9training/Adam/Adam/update_conv2d/kernel/ResourceApplyAdam(1����S@9����S@A����S@I����S@Q���T20?Y.��9@��?�Unknown
�DeviceRandomUniform"9dropout/cond/then/_0/dropout/random_uniform/RandomUniform(1�V-@9�V-@A�V-@I�V-@Qp^�ydv.?YD@�'��?�Unknown
uDeviceMul""dropout/cond/then/_0/dropout/Mul_1(1T㥛� @9T㥛� @AT㥛� @IT㥛� @Qr�K�+?Y��%���?�Unknown
�DeviceBiasAddGrad"@training/Adam/gradients/gradients/dense/BiasAdd_grad/BiasAddGrad(1`��"��@9`��"��@A`��"��@I`��"��@Q�`(��)?Y)��lw��?�Unknown
sDeviceMul" dropout/cond/then/_0/dropout/Mul(1�v��/@9�v��/@A�v��/@I�v��/@Qta�h�t(?Y[˷���?�Unknown
�DeviceResourceApplyAdam"7training/Adam/Adam/update_conv2d/bias/ResourceApplyAdam(1y�&1�@9y�&1�@Ay�&1�@Iy�&1�@Q��/C�(?Y~�O%��?�Unknown
�DeviceGreaterEqual")dropout/cond/then/_0/dropout/GreaterEqual(1��|?5^
@9��|?5^
@A��|?5^
@I��|?5^
@Q4�7��&?Y�!�����?�Unknown
�DeviceAddN"Wtraining/Adam/gradients/gradients/dropout/cond_grad/StatelessIf/then/_15/gradients/AddN(1��|?5^
@9��|?5^
@A��|?5^
@I��|?5^
@Q4�7��&?Yn��7B��?�Unknown
�DeviceResourceApplyAdam"8training/Adam/Adam/update_dense_1/bias/ResourceApplyAdam(1�&1�
@9�&1�
@A�&1�
@I�&1�
@Qt�T���%?Y�{R���?�Unknown
o DeviceArgMax"metrics/accuracy/ArgMax_1(1d;�O��	@9d;�O��	@Ad;�O��	@Id;�O��	@Q?M�r%?Y@�;t���?�Unknown
d!DevicePow"training/Adam/Pow(1��MbX	@9��MbX	@A��MbX	@I��MbX	@Q5��M�<%?Yu�BK��?�Unknown
y"Device_Send"$dropout/cond/then/_0/dropout/Mul/_76(1�E����@9�E����@A�E����@I�E����@Q�>�'�$?Y��3���?�Unknown
�#DeviceMul"itraining/Adam/gradients/gradients/dropout/cond_grad/StatelessIf/then/_15/gradients/dropout/Mul_1_grad/Mul(1�E����@9�E����@A�E����@I�E����@Q�>�'�$?Y�q%���?�Unknown
f$DeviceAddV2"training/Adam/add(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q��'�$?Y;�v�.��?�Unknown
�%Device_Send",dropout/cond/then/_0/OptionalFromValue_1/_78(1R���Q@9R���Q@AR���Q@IR���Q@Q6�!<a$?YX�6�t��?�Unknown
d&DeviceBiasAdd"dense/BiasAdd(1���Mb@9���Mb@A���Mb@I���Mb@Qv?�P*$?YJ�E`���?�Unknown
�'Device_Send",dropout/cond/then/_0/OptionalFromValue_6/_84(1���Mb@9���Mb@A���Mb@I���Mb@Qv?�P*$?Y<�T���?�Unknown
�(DeviceTile"Atraining/Adam/gradients/gradients/loss/dense_1_loss/Sum_grad/Tile(1+���@9+���@A+���@I+���@Q�c\�e�#?Y��;9��?�Unknown
f)DeviceCast"training/Adam/Cast(1����K@9����K@A����K@I����K@Q6�����#?Yq���q��?�Unknown
g*DeviceSum"metrics/accuracy/Sum(1��K7�@9��K7�@A��K7�@I��K7�@Q��օ�"?Y�c����?�Unknown
f+DevicePow"training/Adam/Pow_1(1j�t�@9j�t�@Aj�t�@Ij�t�@Qvd)X�r"?Ye�����?�Unknown
�,Device	ZerosLike".training/Adam/gradients/gradients/zeros_like_2(1j�t�@9j�t�@Aj�t�@Ij�t�@Qvd)X�r"?Y�h���?�Unknown
�-DeviceAssignAddVariableOp"$metrics/accuracy/AssignAddVariableOp(1/�$�@9/�$�@A/�$�@I/�$�@Q��c2""?Y:�9n��?�Unknown
�.DeviceAssignAddVariableOp"&metrics/accuracy/AssignAddVariableOp_1(1/�$�@9/�$�@A/�$�@I/�$�@Q��c2""?Yy�\�.��?�Unknown
�/DeviceMul"gtraining/Adam/gradients/gradients/dropout/cond_grad/StatelessIf/then/_15/gradients/dropout/Mul_grad/Mul(1��K7�A@9��K7�A@A��K7�A@I��K7�A@Q�Ux��!?Y�6D�K��?�Unknown
^0DeviceRelu"
dense/Relu(1#��~j�@9#��~j�@A#��~j�@I#��~j�@Q�һ�``!?Y��S�a��?�Unknown
z1Device_Send"%dropout/cond/then/_0/dropout/Cast/_82(1{�G�z@9{�G�z@A{�G�z@I{�G�z@Q���u)!?YM@�\t��?�Unknown
�2DeviceReluGrad":training/Adam/gradients/gradients/dense/Relu_grad/ReluGrad(1{�G�z@9{�G�z@A{�G�z@I{�G�z@Q���u)!?Y߭���?�Unknown
�3DeviceBiasAddGrad"Btraining/Adam/gradients/gradients/dense_1/BiasAdd_grad/BiasAddGrad(1{�G�z@9{�G�z@A{�G�z@I{�G�z@Q���u)!?Yqo����?�Unknown
k4DeviceEqual"metrics/accuracy/Equal(1���Q�@9���Q�@A���Q�@I���Q�@Qm�l� ?Y�/���?�Unknown
�5DeviceAssignAddVariableOp"&training/Adam/Adam/AssignAddVariableOp(1��ʡE�@9��ʡE�@A��ʡE�@I��ʡE�@Q��0��� ?Ykz=���?�Unknown
f6DeviceBiasAdd"dense_1/BiasAdd(1��~j�t@9��~j�t@A��~j�t@I��~j�t@Q�@N��M ?Y����?�Unknown
o7DeviceDivNoNan"loss/dense_1_loss/value(1333333@9333333@A333333@I333333@Q8�k�� ?Y�������?�Unknown
�8DeviceDivNoNan"Itraining/Adam/gradients/gradients/loss/dense_1_loss/value_grad/div_no_nan(1333333@9333333@A333333@I333333@Q8�k�� ?YW-�����?�Unknown
�9Device_Send"[Func/training/Adam/gradients/gradients/dropout/cond_grad/StatelessIf/then/_15/input/_58/_88(1�l����@9�l����@A�l����@I�l����@Q���?Y�����?�Unknown
�:Device_Send"[Func/training/Adam/gradients/gradients/dropout/cond_grad/StatelessIf/then/_15/input/_62/_86(1�l����@9�l����@A�l����@I�l����@Q���?Yq�S����?�Unknown
�;Device_Send"[Func/training/Adam/gradients/gradients/dropout/cond_grad/StatelessIf/then/_15/input/_63/_90(1�l����@9�l����@A�l����@I�l����@Q���?Y�6����?�Unknown
h<DeviceSum"loss/dense_1_loss/Sum(1V-���@9V-���@AV-���@IV-���@Q�{�9x�?YMש��?�Unknown
�=DeviceMul"^training/Adam/gradients/gradients/loss/dense_1_loss/softmax_cross_entropy_with_logits_grad/mul(1㥛� �@9㥛� �@A㥛� �@I㥛� �@Qo9L�R?Y|0�g���?�Unknown
s>DeviceDivNoNan"metrics/accuracy/div_no_nan(1;�O��n@9;�O��n@A;�O��n@I;�O��n@Q�ˆ�:�?Y�,�����?�Unknown
u?DeviceCast"!dropout/cond/then/_0/dropout/Cast(1�V-@9�V-@A�V-@I�V-@Qp^�ydv?Y���<���?�Unknown
w@DeviceCast"#loss/dense_1_loss/num_elements/Cast(1D�l���@9D�l���@AD�l���@ID�l���@Qq�6.��?Yql�|��?�Unknown
iADeviceCast"metrics/accuracy/Cast(1�p=
ף @9�p=
ף @A�p=
ף @I�p=
ף @Qr� �_�?Yw%�-[��?�Unknown
kBDeviceCast"metrics/accuracy/Cast_1(1�p=
ף @9�p=
ף @A�p=
ף @I�p=
ף @Qr� �_�?Y}ކH:��?�Unknown
�CDevice_Send"�training/Adam/gradients/gradients/dropout/cond_grad/StatelessIf/then/_15/gradients/dropout/Mul_1_grad/Shape_1/OptionalGetValue/_100(1o��ʡ @9o��ʡ @Ao��ʡ @Io��ʡ @Q�����?YH��?�Unknown
�DDevice_Send"�training/Adam/gradients/gradients/dropout/cond_grad/StatelessIf/then/_15/gradients/OptionalFromValue_1_grad/OptionalGetValue/_102(1����Mb @9����Mb @A����Mb @I����Mb @Q�_[q�u?Y�Y����?�Unknown
nEDevice_Recv"metrics/accuracy/Size/_99(1�~j�t��?9�~j�t��?A�~j�t��?I�~j�t��?Q��'�?YA�����?�Unknown
wFDevice_Recv""loss/dense_1_loss/num_elements/_97(1?5^�I�?9?5^�I�?A?5^�I�?I?5^�I�?Q	�>�&?Y�1��:��?�Unknown
�GDevice_Recv"Straining/Adam/gradients/gradients/dropout/cond_grad/StatelessIf/switch_pred/_17/_69(1��~j�t�?9��~j�t�?A��~j�t�?I��~j�t�?Q�@N��M?Y
�Z���?�Unknown
bHDevice_Send"loss/mul/_104(1L7�A`��?9L7�A`��?AL7�A`��?IL7�A`��?Q�:�5Q?Y��ǟ.��?�Unknown
sIDevice_Send"metrics/accuracy/Identity/_106(1L7�A`��?9L7�A`��?AL7�A`��?IL7�A`��?Q�:�5Q?Y<����?�Unknown
xJDeviceUnknown"!_arg_dense_1_target_0_1/_61:_Recv(1y�&1��?9y�&1��?Ay�&1��?Iy�&1��?Q��/C�?Y�������?�Unknown
BKHostIDLE"IDLE1�C�l�=�@A�C�l�=�@a���� �?i���� �?�Unknown
�LHost_Send"�training/Adam/gradients/gradients/dropout/cond_grad/StatelessIf/then/_15/gradients/dropout/Mul_1_grad/Shape_1/OptionalGetValue/_100(1���x��\@9���x��\@A���x��\@I���x��\@a!)'��v�?i�E�~�v�?�Unknown
�MHost_Recv"Otraining/Adam/gradients/gradients/dropout/cond_grad/StatelessIf/pivot_f/_18/_72(1���S�X@9���S�X@A���S�X@I���S�X@a��E�(\�?i�]�"&��?�Unknown
NHost_Send",dropout/cond/then/_0/OptionalFromValue_1/_78(1�x�&1 X@9�x�&1 X@A�x�&1 X@I�x�&1 X@a�뽾��?iNU����?�Unknown
OHost_Send",dropout/cond/then/_0/OptionalFromValue_6/_84(1H�z�W@9H�z�W@AH�z�W@IH�z�W@aw�"D�5�?i�^�L�?�Unknown
wPHost_Recv"$dropout/cond/then/_0/dropout/Mul/_77(1�x�&1�R@9�x�&1�R@A�x�&1�R@I�x�&1�R@aj�"��|?iG%� ��?�Unknown
�QHost_Send"�training/Adam/gradients/gradients/dropout/cond_grad/StatelessIf/then/_15/gradients/OptionalFromValue_1_grad/OptionalGetValue/_102(1=
ףp=J@9=
ףp=J@A=
ףp=J@I=
ףp=J@a���#n�s?iT@f���?�Unknown
xRHost_Recv"%dropout/cond/then/_0/dropout/Cast/_83(1�G�z./@9�G�z./@A�G�z./@I�G�z./@a��I�4W?i�-�驷�?�Unknown
`SHost_Recv"loss/mul/_105(1w��/-@9w��/-@Aw��/-@Iw��/-@a�H�U?i�/j��?�Unknown
qTHost_Recv"metrics/accuracy/Identity/_107(1V-r'@9V-r'@AV-r'@IV-r'@a�-��sQ?iy��8��?�Unknown
�UHost_Recv"[Func/training/Adam/gradients/gradients/dropout/cond_grad/StatelessIf/then/_15/input/_62/_87(1\���(\%@9\���(\%@A\���(\%@I\���(\%@a/R�P�O?i��+��?�Unknown
�VHost_Recv"[Func/training/Adam/gradients/gradients/dropout/cond_grad/StatelessIf/then/_15/input/_63/_91(1��K7�A#@9��K7�A#@A��K7�A#@I��K7�A#@a1ц��L?i�w�/V��?�Unknown
�WHost_Recv"[Func/training/Adam/gradients/gradients/dropout/cond_grad/StatelessIf/then/_15/input/_58/_89(1�MbX�!@9�MbX�!@A�MbX�!@I�MbX�!@ac^{&�J?i�:t��?�Unknown
�XHostOptionalFromValue"(dropout/cond/then/_0/OptionalFromValue_1(1�E���T@9�E���T@A�E���T@I�E���T@a�k�\|WD?i(8QS��?�Unknown
�YHost	_HostSend"otraining/Adam/gradients/gradients/dropout/cond_grad/StatelessIf/then/_15/gradients/dropout/Mul_1_grad/Shape/_92(1��v���@9��v���@A��v���@I��v���@a��m3�A?i%y,����?�Unknown
�ZHost_Send"Straining/Adam/gradients/gradients/dropout/cond_grad/StatelessIf/switch_pred/_17/_66(1�G�z@9�G�z@A�G�z@I�G�z@au%}�`??i���l��?�Unknown
�[HostOptionalFromValue"(dropout/cond/then/_0/OptionalFromValue_6(1
ףp=
@9
ףp=
@A
ףp=
@I
ףp=
@a��A(S�=?i��q>'��?�Unknown
�\Host_Send"Straining/Adam/gradients/gradients/dropout/cond_grad/StatelessIf/switch_pred/_17/_68(1-���F@9-���F@A-���F@I-���F@a���3�<?i���d���?�Unknown
�]Host	_HostSend"qtraining/Adam/gradients/gradients/dropout/cond_grad/StatelessIf/then/_15/gradients/dropout/Mul_1_grad/Shape_1/_94(1u�V�@9u�V�@Au�V�@Iu�V�@a�BO�!:?i������?�Unknown
�^HostOptionalGetValue"~training/Adam/gradients/gradients/dropout/cond_grad/StatelessIf/then/_15/gradients/dropout/Mul_1_grad/Shape_1/OptionalGetValue(1+���@9+���@A+���@I+���@a�>H:-�4?i�����?�Unknown
�_HostOptionalGetValue"|training/Adam/gradients/gradients/dropout/cond_grad/StatelessIf/then/_15/gradients/dropout/Mul_1_grad/Shape/OptionalGetValue(1�ʡE��@9�ʡE��@A�ʡE��@I�ʡE��@a�J���*?i���nC��?�Unknown
�`HostOptionalGetValue"|training/Adam/gradients/gradients/dropout/cond_grad/StatelessIf/then/_15/gradients/OptionalFromValue_1_grad/OptionalGetValue(1Zd;�O@9Zd;�O@AZd;�O@IZd;�O@aT�$��)?i�E�����?�Unknown
zaHost_Send"'dropout/cond/else/_1/OptionalNone_6/_80(1h��|?5�?9h��|?5�?Ah��|?5�?Ih��|?5�?a	 �{G"?i     �?�Unknown*�k
�DeviceResourceApplyAdam"8training/Adam/Adam/update_dense/kernel/ResourceApplyAdam(1��S��@�@9��S��@�@A��S��@�@I��S��@�@Q��E����?Y��E����?�Unknown
�DeviceMatMul":training/Adam/gradients/gradients/dense/MatMul_grad/MatMul(1�x�&�d�@9�x�&�d�@A�x�&�d�@I�x�&�d�@Q�����-�?Y�cy�	�?�Unknown
bDeviceMatMul"dense/MatMul(1��"����@9��"����@A��"����@I��"����@Q(&0'�?Y�(~���?�Unknown
�DeviceMatMul"<training/Adam/gradients/gradients/dense/MatMul_grad/MatMul_1(1j�t����@9j�t����@Aj�t����@Ij�t����@Q�%U��?Y��"_��?�Unknown
�DeviceMaxPoolGrad"Htraining/Adam/gradients/gradients/max_pooling2d/MaxPool_grad/MaxPoolGrad(1��/�Pd@9��/�Pd@A��/�Pd@I��/�Pd@Q�:(d��?Yr�����?�Unknown
�DeviceConv2DBackpropFilter"Itraining/Adam/gradients/gradients/conv2d/Conv2D_grad/Conv2DBackpropFilter(1�I+�X@9�I+�X@A�I+�X@I�I+�X@Q���:w?Y�<5Gc�?�Unknown
�DeviceReluGrad";training/Adam/gradients/gradients/conv2d/Relu_grad/ReluGrad(1���MbhW@9���MbhW@A���MbhW@I���MbhW@Qd�jY��u?Y��<�?�Unknown
gDevice_FusedConv2D"conv2d/Relu(1^�I�Q@9^�I�Q@A^�I�Q@I^�I�Q@Q+���Zp?Y���p�\�?�Unknown
�	DeviceBiasAddGrad"Atraining/Adam/gradients/gradients/conv2d/BiasAdd_grad/BiasAddGrad(1�K7�A`G@9�K7�A`G@A�K7�A`G@I�K7�A`G@Q^ʏ��e?Y�i�|�r�?�Unknown
l
DeviceMaxPool"max_pooling2d/MaxPool(1�~j�tCE@9�~j�tCE@A�~j�tCE@I�~j�tCE@Q�sHL��c?Y�Gs���?�Unknown
�DeviceSoftmaxCrossEntropyWithLogits"3loss/dense_1_loss/softmax_cross_entropy_with_logits(1��Q��;@9��Q��;@A��Q��;@I��Q��;@Q ���ٳY?Y�0`]��?�Unknown
dDeviceMatMul"dense_1/MatMul(1�Zd;�3@9�Zd;�3@A�Zd;�3@I�Zd;�3@Q1S]C�R?Y������?�Unknown
�Device	Transpose"ntraining/Adam/gradients/gradients/max_pooling2d/MaxPool_grad/MaxPoolGrad-2-TransposeNHWCToNCHW-LayoutOptimizer(1��ʡE1@9��ʡE1@A��ʡE1@I��ʡE1@Q��(���O?Y�i�����?�Unknown
�Device	Transpose"=max_pooling2d/MaxPool-0-0-TransposeNCHWToNHWC-LayoutOptimizer(1�K7�A`-@9�K7�A`-@A�K7�A`-@I�K7�A`-@Q�o]n�oK?Y>h�{��?�Unknown
vDeviceUnknown"_arg_conv2d_input_0_0/_59:_Recv(1���K�#@9���K�#@A���K�#@I���K�#@Q��f
��B?Y���	%��?�Unknown
fDeviceSoftmax"dense_1/Softmax(1Zd;� @9Zd;� @AZd;� @IZd;� @Qbp^�� >?Y����?�Unknown
�DeviceResourceApplyAdam"6training/Adam/Adam/update_dense/bias/ResourceApplyAdam(1#��~j�@9#��~j�@A#��~j�@I#��~j�@Qm� �]3?Y���P��?�Unknown
�DeviceMatMul">training/Adam/gradients/gradients/dense_1/MatMul_grad/MatMul_1(1j�t�@9j�t�@Aj�t�@Ij�t�@Q��~��2?Y�)s���?�Unknown
�DeviceResourceApplyAdam":training/Adam/Adam/update_dense_1/kernel/ResourceApplyAdam(1d;�O�@9d;�O�@Ad;�O�@Id;�O�@Q�ř<��2?Y(�����?�Unknown
�DeviceMatMul"<training/Adam/gradients/gradients/dense_1/MatMul_grad/MatMul(1�t��@9�t��@A�t��@I�t��@Qr�N��2?Y��R��?�Unknown
mDeviceArgMax"metrics/accuracy/ArgMax(1/�$��@9/�$��@A/�$��@I/�$��@Q<��gJ2?Y �����?�Unknown
�DeviceResourceApplyAdam"9training/Adam/Adam/update_conv2d/kernel/ResourceApplyAdam(1����S@9����S@A����S@I����S@Q��e22?Yߛ����?�Unknown
�DeviceRandomUniform"9dropout/cond/then/_0/dropout/random_uniform/RandomUniform(1�V-@9�V-@A�V-@I�V-@Q5����0?Yon����?�Unknown
uDeviceMul""dropout/cond/then/_0/dropout/Mul_1(1T㥛� @9T㥛� @AT㥛� @IT㥛� @Qr�22 .?YD�����?�Unknown
�DeviceBiasAddGrad"@training/Adam/gradients/gradients/dense/BiasAdd_grad/BiasAddGrad(1`��"��@9`��"��@A`��"��@I`��"��@Q�$r(�,?Y#�խ��?�Unknown
sDeviceMul" dropout/cond/then/_0/dropout/Mul(1�v��/@9�v��/@A�v��/@I�v��/@Q61e��A+?Yv�J�a��?�Unknown
�DeviceResourceApplyAdam"7training/Adam/Adam/update_conv2d/bias/ResourceApplyAdam(1y�&1�@9y�&1�@Ay�&1�@Iy�&1�@Q�4�I�*?Yi��d��?�Unknown
�DeviceGreaterEqual")dropout/cond/then/_0/dropout/GreaterEqual(1��|?5^
@9��|?5^
@A��|?5^
@I��|?5^
@Q*E,j�(?Y-�|k���?�Unknown
�DeviceAddN"Wtraining/Adam/gradients/gradients/dropout/cond_grad/StatelessIf/then/_15/gradients/AddN(1��|?5^
@9��|?5^
@A��|?5^
@I��|?5^
@Q*E,j�(?Y�sr"��?�Unknown
�DeviceResourceApplyAdam"8training/Adam/Adam/update_dense_1/bias/ResourceApplyAdam(1�&1�
@9�&1�
@A�&1�
@I�&1�
@Q�F��4c(?Y�j����?�Unknown
oDeviceArgMax"metrics/accuracy/ArgMax_1(1d;�O��	@9d;�O��	@Ad;�O��	@Id;�O��	@Q����'?Y*v'��?�Unknown
d DevicePow"training/Adam/Pow(1��MbX	@9��MbX	@A��MbX	@I��MbX	@QlL (��'?Y��̡��?�Unknown
y!Device_Send"$dropout/cond/then/_0/dropout/Mul/_76(1�E����@9�E����@A�E����@I�E����@QPj�*1'?Y� t���?�Unknown
�"DeviceMul"itraining/Adam/gradients/gradients/dropout/cond_grad/StatelessIf/then/_15/gradients/dropout/Mul_1_grad/Mul(1�E����@9�E����@A�E����@I�E����@QPj�*1'?YcW���?�Unknown
f#DeviceAddV2"training/Adam/add(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q�Q|��&?YXw1���?�Unknown
�$Device_Send",dropout/cond/then/_0/OptionalFromValue_1/_78(1R���Q@9R���Q@AR���Q@IR���Q@Q�S�B��&?Y�F{�b��?�Unknown
d%DeviceBiasAdd"dense/BiasAdd(1���Mb@9���Mb@A���Mb@I���Mb@Q}U�	�y&?Y2�+6���?�Unknown
�&Device_Send",dropout/cond/then/_0/OptionalFromValue_6/_84(1���Mb@9���Mb@A���Mb@I���Mb@Q}U�	�y&?Y�w��1��?�Unknown
�'DeviceTile"Atraining/Adam/gradients/gradients/loss/dense_1_loss/Sum_grad/Tile(1+���@9+���@A+���@I+���@QNW>�U<&?Y�{9����?�Unknown
f(DeviceCast"training/Adam/Cast(1����K@9����K@A����K@I����K@Q�Z�]��%?Y2V����?�Unknown
g)DeviceSum"metrics/accuracy/Sum(1��K7�@9��K7�@A��K7�@I��K7�@QR��[5%?YNEvB��?�Unknown
f*DevicePow"training/Adam/Pow_1(1j�t�@9j�t�@Aj�t�@Ij�t�@Q d1?�$?YdYt���?�Unknown
�+Device	ZerosLike".training/Adam/gradients/gradients/zeros_like_2(1j�t�@9j�t�@Aj�t�@Ij�t�@Q d1?�$?Yz�lr���?�Unknown
�,DeviceAssignAddVariableOp"$metrics/accuracy/AssignAddVariableOp(1/�$�@9/�$�@A/�$�@I/�$�@Q�g��v$?Y0�����?�Unknown
�-DeviceAssignAddVariableOp"&metrics/accuracy/AssignAddVariableOp_1(1/�$�@9/�$�@A/�$�@I/�$�@Q�g��v$?Y�F!W��?�Unknown
�.DeviceMul"gtraining/Adam/gradients/gradients/dropout/cond_grad/StatelessIf/then/_15/gradients/dropout/Mul_grad/Mul(1��K7�A@9��K7�A@A��K7�A@I��K7�A@Qc�=+�#?Y�Z�Ô��?�Unknown
^/DeviceRelu"
dense/Relu(1#��~j�@9#��~j�@A#��~j�@I#��~j�@Qm� �]#?Y9fl����?�Unknown
z0Device_Send"%dropout/cond/then/_0/dropout/Cast/_82(1{�G�z@9{�G�z@A{�G�z@I{�G�z@Q�no� #?Y0݊����?�Unknown
�1DeviceReluGrad":training/Adam/gradients/gradients/dense/Relu_grad/ReluGrad(1{�G�z@9{�G�z@A{�G�z@I{�G�z@Q�no� #?Y'T��.��?�Unknown
�2DeviceBiasAddGrad"Btraining/Adam/gradients/gradients/dense_1/BiasAdd_grad/BiasAddGrad(1{�G�z@9{�G�z@A{�G�z@I{�G�z@Q�no� #?Y�ǿ`��?�Unknown
k3DeviceEqual"metrics/accuracy/Equal(1���Q�@9���Q�@A���Q�@I���Q�@QE�X��j"?Y� �n���?�Unknown
�4DeviceAssignAddVariableOp"&training/Adam/Adam/AssignAddVariableOp(1��ʡE�@9��ʡE�@A��ʡE�@I��ʡE�@QTt�;i"?Y�٩����?�Unknown
f5DeviceBiasAdd"dense_1/BiasAdd(1��~j�t@9��~j�t@A��~j�t@I��~j�t@Q$vC�+"?Y��y����?�Unknown
o6DeviceDivNoNan"loss/dense_1_loss/value(1333333@9333333@A333333@I333333@Q�w�ȗ�!?YP������?�Unknown
�7DeviceDivNoNan"Itraining/Adam/gradients/gradients/loss/dense_1_loss/value_grad/div_no_nan(1333333@9333333@A333333@I333333@Q�w�ȗ�!?Y�s���?�Unknown
�8Device_Send"[Func/training/Adam/gradients/gradients/dropout/cond_grad/StatelessIf/then/_15/input/_58/_88(1�l����@9�l����@A�l����@I�l����@Q�y��b�!?Y���)��?�Unknown
�9Device_Send"[Func/training/Adam/gradients/gradients/dropout/cond_grad/StatelessIf/then/_15/input/_62/_86(1�l����@9�l����@A�l����@I�l����@Q�y��b�!?Y�źD��?�Unknown
�:Device_Send"[Func/training/Adam/gradients/gradients/dropout/cond_grad/StatelessIf/then/_15/input/_63/_90(1�l����@9�l����@A�l����@I�l����@Q�y��b�!?Y_��_��?�Unknown
h;DeviceSum"loss/dense_1_loss/Sum(1V-���@9V-���@AV-���@IV-���@Q�!��x�!?Y�k|�z��?�Unknown
�<DeviceMul"^training/Adam/gradients/gradients/loss/dense_1_loss/softmax_cross_entropy_with_logits_grad/mul(1㥛� �@9㥛� �@A㥛� �@I㥛� �@Q�{bV-t!?Y��Q���?�Unknown
s=DeviceDivNoNan"metrics/accuracy/div_no_nan(1;�O��n@9;�O��n@A;�O��n@I;�O��n@Qe}�6!?Y1��z���?�Unknown
u>DeviceCast"!dropout/cond/then/_0/dropout/Cast(1�V-@9�V-@A�V-@I�V-@Q5���� ?Y�����?�Unknown
w?DeviceCast"#loss/dense_1_loss/num_elements/Cast(1D�l���@9D�l���@AD�l���@ID�l���@Qׂ6qX ?Ya����?�Unknown
i@DeviceCast"metrics/accuracy/Cast(1�p=
ף @9�p=
ף @A�p=
ף @I�p=
ף @Q1?Y
������?�Unknown
kADeviceCast"metrics/accuracy/Cast_1(1�p=
ף @9�p=
ף @A�p=
ף @I�p=
ף @Q1?Y�t�\���?�Unknown
�BDevice_Send"�training/Adam/gradients/gradients/dropout/cond_grad/StatelessIf/then/_15/gradients/dropout/Mul_1_grad/Shape_1/OptionalGetValue/_100(1o��ʡ @9o��ʡ @Ao��ʡ @Io��ʡ @QNd��3?Y������?�Unknown
�CDevice_Send"�training/Adam/gradients/gradients/dropout/cond_grad/StatelessIf/then/_15/gradients/OptionalFromValue_1_grad/OptionalGetValue/_102(1����Mb @9����Mb @A����Mb @I����Mb @Q����?Y��}����?�Unknown
nDDevice_Recv"metrics/accuracy/Size/_99(1�~j�t��?9�~j�t��?A�~j�t��?I�~j�t��?Q�Q|��?Y��)[S��?�Unknown
wEDevice_Recv""loss/dense_1_loss/num_elements/_97(1?5^�I�?9?5^�I�?A?5^�I�?I?5^�I�?Q�����u?YOU���?�Unknown
�FDevice_Recv"Straining/Adam/gradients/gradients/dropout/cond_grad/StatelessIf/switch_pred/_17/_69(1��~j�t�?9��~j�t�?A��~j�t�?I��~j�t�?Q$vC�+?YkgOg���?�Unknown
bGDevice_Send"loss/mul/_104(1L7�A`��?9L7�A`��?AL7�A`��?IL7�A`��?Q���q�?Y����?�Unknown
sHDevice_Send"metrics/accuracy/Identity/_106(1L7�A`��?9L7�A`��?AL7�A`��?IL7�A`��?Q���q�?Yü����?�Unknown
xIDeviceUnknown"!_arg_dense_1_target_0_1/_61:_Recv(1y�&1��?9y�&1��?Ay�&1��?Iy�&1��?Q�4�I�
?Y      �?�Unknown
�JHost_Send"�training/Adam/gradients/gradients/dropout/cond_grad/StatelessIf/then/_15/gradients/dropout/Mul_1_grad/Shape_1/OptionalGetValue/_100(1���x��\@9���x��\@A���x��\@I���x��\@a�y!�w��?i�y!�w��?�Unknown
�KHost_Recv"Otraining/Adam/gradients/gradients/dropout/cond_grad/StatelessIf/pivot_f/_18/_72(1���S�X@9���S�X@A���S�X@I���S�X@a��&q��?i� �g�F�?�Unknown
LHost_Send",dropout/cond/then/_0/OptionalFromValue_1/_78(1�x�&1 X@9�x�&1 X@A�x�&1 X@I�x�&1 X@a��v3�/�?i�e_����?�Unknown
MHost_Send",dropout/cond/then/_0/OptionalFromValue_6/_84(1H�z�W@9H�z�W@AH�z�W@IH�z�W@a�����c�?i}&mh_�?�Unknown
wNHost_Recv"$dropout/cond/then/_0/dropout/Mul/_77(1�x�&1�R@9�x�&1�R@A�x�&1�R@I�x�&1�R@a��7Q��?i��R��?�Unknown
�OHost_Send"�training/Adam/gradients/gradients/dropout/cond_grad/StatelessIf/then/_15/gradients/OptionalFromValue_1_grad/OptionalGetValue/_102(1=
ףp=J@9=
ףp=J@A=
ףp=J@I=
ףp=J@aۂÄ;޴?iL�,��d�?�Unknown
xPHost_Recv"%dropout/cond/then/_0/dropout/Cast/_83(1�G�z./@9�G�z./@A�G�z./@I�G�z./@a�W��˘?i
��B+�?�Unknown
`QHost_Recv"loss/mul/_105(1w��/-@9w��/-@Aw��/-@Iw��/-@a��}�Z'�?iG�_}��?�Unknown
qRHost_Recv"metrics/accuracy/Identity/_107(1V-r'@9V-r'@AV-r'@IV-r'@a�/&f��?i� _��y�?�Unknown
�SHost_Recv"[Func/training/Adam/gradients/gradients/dropout/cond_grad/StatelessIf/then/_15/input/_62/_87(1\���(\%@9\���(\%@A\���(\%@I\���(\%@a3�!���?i]�n9��?�Unknown
�THost_Recv"[Func/training/Adam/gradients/gradients/dropout/cond_grad/StatelessIf/then/_15/input/_63/_91(1��K7�A#@9��K7�A#@A��K7�A#@I��K7�A#@aٶ����?i8ż	|�?�Unknown
�UHost_Recv"[Func/training/Adam/gradients/gradients/dropout/cond_grad/StatelessIf/then/_15/input/_58/_89(1�MbX�!@9�MbX�!@A�MbX�!@I�MbX�!@a#�����?i���k��?�Unknown
�VHostOptionalFromValue"(dropout/cond/then/_0/OptionalFromValue_1(1�E���T@9�E���T@A�E���T@I�E���T@a|��֍��?i�go	^E�?�Unknown
�WHost	_HostSend"otraining/Adam/gradients/gradients/dropout/cond_grad/StatelessIf/then/_15/gradients/dropout/Mul_1_grad/Shape/_92(1��v���@9��v���@A��v���@I��v���@a�AE�ł?i�|,Mt��?�Unknown
�XHost_Send"Straining/Adam/gradients/gradients/dropout/cond_grad/StatelessIf/switch_pred/_17/_66(1�G�z@9�G�z@A�G�z@I�G�z@a�ň��À?i��
���?�Unknown
�YHostOptionalFromValue"(dropout/cond/then/_0/OptionalFromValue_6(1
ףp=
@9
ףp=
@A
ףp=
@I
ףp=
@ai�'4��?i �r�B�?�Unknown
�ZHost_Send"Straining/Adam/gradients/gradients/dropout/cond_grad/StatelessIf/switch_pred/_17/_68(1-���F@9-���F@A-���F@I-���F@a���ب~?i)&��P�?�Unknown
�[Host	_HostSend"qtraining/Adam/gradients/gradients/dropout/cond_grad/StatelessIf/then/_15/gradients/dropout/Mul_1_grad/Shape_1/_94(1u�V�@9u�V�@Au�V�@Iu�V�@a#<�P~�{?i��Ǩm��?�Unknown
�\HostOptionalGetValue"~training/Adam/gradients/gradients/dropout/cond_grad/StatelessIf/then/_15/gradients/dropout/Mul_1_grad/Shape_1/OptionalGetValue(1+���@9+���@A+���@I+���@alN\�vv?i4x����?�Unknown
�]HostOptionalGetValue"|training/Adam/gradients/gradients/dropout/cond_grad/StatelessIf/then/_15/gradients/dropout/Mul_1_grad/Shape/OptionalGetValue(1�ʡE��@9�ʡE��@A�ʡE��@I�ʡE��@a��ռ�l?i��R6��?�Unknown
�^HostOptionalGetValue"|training/Adam/gradients/gradients/dropout/cond_grad/StatelessIf/then/_15/gradients/OptionalFromValue_1_grad/OptionalGetValue(1Zd;�O@9Zd;�O@AZd;�O@IZd;�O@a�͵$�k?i��v���?�Unknown
z_Host_Send"'dropout/cond/else/_1/OptionalNone_6/_80(1h��|?5�?9h��|?5�?Ah��|?5�?Ih��|?5�?a�Q�@c?i      �?�Unknown2Nvidia GPU (Turing)