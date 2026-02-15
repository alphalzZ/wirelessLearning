classdef Shape_To_ReshapeLayer1045 < nnet.layer.Layer & nnet.layer.Formattable
    % A custom layer auto-generated while importing an ONNX network.

    %#codegen
    %#ok<*PROPLC>
    %#ok<*NBRAK>
    %#ok<*INUSL>
    %#ok<*VARARG>

    properties (Learnable)
        const_fold_opt__5883
        n_nrecevier_mod_1671
        n_nrecevier_mod_1673
    end

    properties
        ONNXParams         % An ONNXParameters object containing parameters used by this layer.
    end

    methods
        function this = Shape_To_ReshapeLayer1045(name, onnxParams)
            this.Name = name;
            this.NumInputs = 2;
            this.NumOutputs = 5;
            this.OutputNames = {'n_nrecevier_mod_1808', 'n_nrecevier_mod_1800', 'n_nrecevier_mod_1799', 'n_nrecevier_mod_1800NumDims', 'n_nrecevier_mod_1799NumDims'};
            this.ONNXParams = onnxParams;
            this.const_fold_opt__5883 = onnxParams.Learnables.const_fold_opt__5883;
            this.n_nrecevier_mod_1671 = onnxParams.Learnables.n_nrecevier_mod_1671;
            this.n_nrecevier_mod_1673 = onnxParams.Learnables.n_nrecevier_mod_1673;
        end

        function [n_nrecevier_mod_1808, n_nrecevier_mod_1800, n_nrecevier_mod_1799, n_nrecevier_mod_1800NumDims, n_nrecevier_mod_1799NumDims] = predict(this, n_nrecevier_mod_1703, Transpose__5247_0)
            if isdlarray(n_nrecevier_mod_1703)
                n_nrecevier_mod_1703 = stripdims(n_nrecevier_mod_1703);
            end
            if isdlarray(Transpose__5247_0)
                Transpose__5247_0 = stripdims(Transpose__5247_0);
            end
            n_nrecevier_mod_1703NumDims = 4;
            Transpose__5247_0NumDims = 4;
            onnxParams = this.ONNXParams;
            onnxParams.Learnables.const_fold_opt__5883 = this.const_fold_opt__5883;
            onnxParams.Learnables.n_nrecevier_mod_1671 = this.n_nrecevier_mod_1671;
            onnxParams.Learnables.n_nrecevier_mod_1673 = this.n_nrecevier_mod_1673;
            [n_nrecevier_mod_1808, n_nrecevier_mod_1800, n_nrecevier_mod_1799, n_nrecevier_mod_1808NumDims, n_nrecevier_mod_1800NumDims, n_nrecevier_mod_1799NumDims] = Shape_To_ReshapeFcn(n_nrecevier_mod_1703, Transpose__5247_0, n_nrecevier_mod_1703NumDims, Transpose__5247_0NumDims, onnxParams, 'Training', false, ...
                'InputDataPermutation', {[4 3 1 2], [4 1 2 3], ['as-is'], ['as-is']}, ...
                'OutputDataPermutation', {[2 3 4 1], ['as-is'], ['as-is'], ['as-is'], ['as-is'], ['as-is']});
            if any(cellfun(@(A)~isnumeric(A) && ~islogical(A), {n_nrecevier_mod_1808, n_nrecevier_mod_1800, n_nrecevier_mod_1799}))
                fprintf('Runtime error in network. At least one output of custom layer ''%s'' is a non-numeric, non-logical value.\n', 'Shape_To_ReshapeLayer1045');
                error(message('nnet_cnn_onnx:onnx:BadCustomLayerRuntimeOutput', 'Shape_To_ReshapeLayer1045'));
            end
            n_nrecevier_mod_1808 = dlarray(single(n_nrecevier_mod_1808), 'SSCB');
            n_nrecevier_mod_1800 = dlarray(single(n_nrecevier_mod_1800), repmat('U', 1, max(2, n_nrecevier_mod_1800NumDims)));
            n_nrecevier_mod_1799 = dlarray(single(n_nrecevier_mod_1799), repmat('U', 1, max(2, n_nrecevier_mod_1799NumDims)));
            n_nrecevier_mod_1800NumDims = dlarray(ones(1,n_nrecevier_mod_1800NumDims,'like',n_nrecevier_mod_1808), 'UU');
            n_nrecevier_mod_1799NumDims = dlarray(ones(1,n_nrecevier_mod_1799NumDims,'like',n_nrecevier_mod_1808), 'UU');
            if ~coder.target('MATLAB')
                n_nrecevier_mod_1808 = extractdata(n_nrecevier_mod_1808);
                n_nrecevier_mod_1800 = extractdata(n_nrecevier_mod_1800);
                n_nrecevier_mod_1799 = extractdata(n_nrecevier_mod_1799);
                n_nrecevier_mod_1800NumDims = extractdata(n_nrecevier_mod_1800NumDims);
                n_nrecevier_mod_1799NumDims = extractdata(n_nrecevier_mod_1799NumDims);
            end
        end

        function [n_nrecevier_mod_1808, n_nrecevier_mod_1800, n_nrecevier_mod_1799, n_nrecevier_mod_1800NumDims, n_nrecevier_mod_1799NumDims] = forward(this, n_nrecevier_mod_1703, Transpose__5247_0)
            if isdlarray(n_nrecevier_mod_1703)
                n_nrecevier_mod_1703 = stripdims(n_nrecevier_mod_1703);
            end
            if isdlarray(Transpose__5247_0)
                Transpose__5247_0 = stripdims(Transpose__5247_0);
            end
            n_nrecevier_mod_1703NumDims = 4;
            Transpose__5247_0NumDims = 4;
            onnxParams = this.ONNXParams;
            onnxParams.Learnables.const_fold_opt__5883 = this.const_fold_opt__5883;
            onnxParams.Learnables.n_nrecevier_mod_1671 = this.n_nrecevier_mod_1671;
            onnxParams.Learnables.n_nrecevier_mod_1673 = this.n_nrecevier_mod_1673;
            [n_nrecevier_mod_1808, n_nrecevier_mod_1800, n_nrecevier_mod_1799, n_nrecevier_mod_1808NumDims, n_nrecevier_mod_1800NumDims, n_nrecevier_mod_1799NumDims] = Shape_To_ReshapeFcn(n_nrecevier_mod_1703, Transpose__5247_0, n_nrecevier_mod_1703NumDims, Transpose__5247_0NumDims, onnxParams, 'Training', true, ...
                'InputDataPermutation', {[4 3 1 2], [4 1 2 3], ['as-is'], ['as-is']}, ...
                'OutputDataPermutation', {[2 3 4 1], ['as-is'], ['as-is'], ['as-is'], ['as-is'], ['as-is']});
            if any(cellfun(@(A)~isnumeric(A) && ~islogical(A), {n_nrecevier_mod_1808, n_nrecevier_mod_1800, n_nrecevier_mod_1799}))
                fprintf('Runtime error in network. At least one output of custom layer ''%s'' is a non-numeric, non-logical value.\n', 'Shape_To_ReshapeLayer1045');
                error(message('nnet_cnn_onnx:onnx:BadCustomLayerRuntimeOutput', 'Shape_To_ReshapeLayer1045'));
            end
            n_nrecevier_mod_1808 = dlarray(single(n_nrecevier_mod_1808), 'SSCB');
            n_nrecevier_mod_1800 = dlarray(single(n_nrecevier_mod_1800), repmat('U', 1, max(2, n_nrecevier_mod_1800NumDims)));
            n_nrecevier_mod_1799 = dlarray(single(n_nrecevier_mod_1799), repmat('U', 1, max(2, n_nrecevier_mod_1799NumDims)));
            n_nrecevier_mod_1800NumDims = dlarray(ones(1,n_nrecevier_mod_1800NumDims,'like',n_nrecevier_mod_1808), 'UU');
            n_nrecevier_mod_1799NumDims = dlarray(ones(1,n_nrecevier_mod_1799NumDims,'like',n_nrecevier_mod_1808), 'UU');
            if ~coder.target('MATLAB')
                n_nrecevier_mod_1808 = extractdata(n_nrecevier_mod_1808);
                n_nrecevier_mod_1800 = extractdata(n_nrecevier_mod_1800);
                n_nrecevier_mod_1799 = extractdata(n_nrecevier_mod_1799);
                n_nrecevier_mod_1800NumDims = extractdata(n_nrecevier_mod_1800NumDims);
                n_nrecevier_mod_1799NumDims = extractdata(n_nrecevier_mod_1799NumDims);
            end
        end
    end
end

function [n_nrecevier_mod_1808, n_nrecevier_mod_1800, n_nrecevier_mod_1799, n_nrecevier_mod_1808NumDims, n_nrecevier_mod_1800NumDims, n_nrecevier_mod_1799NumDims, state] = Shape_To_ReshapeFcn(n_nrecevier_mod_1703, Transpose__5247_0, n_nrecevier_mod_1703NumDims, Transpose__5247_0NumDims, params, varargin)
%SHAPE_TO_RESHAPEFCN Function implementing an imported ONNX network.
%
% THIS FILE WAS AUTO-GENERATED BY importONNXFunction.
% ONNX Operator Set Version: 18
%
% Variable names in this function are taken from the original ONNX file.
%
% [N_NRECEVIER_MOD_1808, N_NRECEVIER_MOD_1800, N_NRECEVIER_MOD_1799] = Shape_To_ReshapeFcn(N_NRECEVIER_MOD_1703, TRANSPOSE__5247_0, PARAMS)
%			- Evaluates the imported ONNX network SHAPE_TO_RESHAPEFCN with input(s)
%			N_NRECEVIER_MOD_1703, TRANSPOSE__5247_0 and the imported network parameters in PARAMS. Returns
%			network output(s) in N_NRECEVIER_MOD_1808, N_NRECEVIER_MOD_1800, N_NRECEVIER_MOD_1799.
%
% [N_NRECEVIER_MOD_1808, N_NRECEVIER_MOD_1800, N_NRECEVIER_MOD_1799, STATE] = Shape_To_ReshapeFcn(N_NRECEVIER_MOD_1703, TRANSPOSE__5247_0, PARAMS)
%			- Additionally returns state variables in STATE. When training,
%			use this form and set TRAINING to true.
%
% [__] = Shape_To_ReshapeFcn(N_NRECEVIER_MOD_1703, TRANSPOSE__5247_0, PARAMS, 'NAME1', VAL1, 'NAME2', VAL2, ...)
%			- Specifies additional name-value pairs described below:
%
% 'Training'
% 			Boolean indicating whether the network is being evaluated for
%			prediction or training. If TRAINING is true, state variables
%			will be updated.
%
% 'InputDataPermutation'
%			'auto' - Automatically attempt to determine the permutation
%			 between the dimensions of the input data and the dimensions of
%			the ONNX model input. For example, the permutation from HWCN
%			(MATLAB standard) to NCHW (ONNX standard) uses the vector
%			[4 3 1 2]. See the documentation for IMPORTONNXFUNCTION for
%			more information about automatic permutation.
%
%			'none' - Input(s) are passed in the ONNX model format. See 'Inputs'.
%
%			numeric vector - The permutation vector describing the
%			transformation between input data dimensions and the expected
%			ONNX input dimensions.%
%			cell array - If the network has multiple inputs, each cell
%			contains 'auto', 'none', or a numeric vector.
%
% 'OutputDataPermutation'
%			'auto' - Automatically attempt to determine the permutation
%			between the dimensions of the output and a conventional MATLAB
%			dimension ordering. For example, the permutation from NC (ONNX
%			standard) to CN (MATLAB standard) uses the vector [2 1]. See
%			the documentation for IMPORTONNXFUNCTION for more information
%			about automatic permutation.
%
%			'none' - Return output(s) as given by the ONNX model. See 'Outputs'.
%
%			numeric vector - The permutation vector describing the
%			transformation between the ONNX output dimensions and the
%			desired output dimensions.%
%			cell array - If the network has multiple outputs, each cell
%			contains 'auto', 'none' or a numeric vector.
%
% Inputs:
% -------
% N_NRECEVIER_MOD_1703, TRANSPOSE__5247_0
%			- Input(s) to the ONNX network.
%			  The input size(s) expected by the ONNX file are:
%				  N_NRECEVIER_MOD_1703:		[Unknown, Unknown, Unknown, Unknown]				Type: FLOAT
%				  TRANSPOSE__5247_0:		[Unknown, Unknown, Unknown, Unknown]				Type: FLOAT
%			  By default, the function will try to permute the input(s)
%			  into this dimension ordering. If the default is incorrect,
%			  use the 'InputDataPermutation' argument to control the
%			  permutation.
%
%
% PARAMS	- Network parameters returned by 'importONNXFunction'.
%
%
% Outputs:
% --------
% N_NRECEVIER_MOD_1808, N_NRECEVIER_MOD_1800, N_NRECEVIER_MOD_1799
%			- Output(s) of the ONNX network.
%			  Without permutation, the size(s) of the outputs are:
%				  N_NRECEVIER_MOD_1808:		[Unknown, Unknown, Unknown, Unknown]				Type: FLOAT
%				  N_NRECEVIER_MOD_1800:		[1, 1]				Type: FLOAT
%				  N_NRECEVIER_MOD_1799:		[1, 1]				Type: FLOAT
%			  By default, the function will try to permute the output(s)
%			  from this dimension ordering into a conventional MATLAB
%			  ordering. If the default is incorrect, use the
%			  'OutputDataPermutation' argument to control the permutation.
%
% STATE		- (Optional) State variables. When TRAINING is true, these will
% 			  have been updated from the original values in PARAMS.State.
%
%
%  See also importONNXFunction

% Preprocess the input data and arguments:
[n_nrecevier_mod_1703, Transpose__5247_0, Training, outputDataPerms, anyDlarrayInputs] = preprocessInput(n_nrecevier_mod_1703, Transpose__5247_0, params, varargin{:});
% Put all variables into a single struct to implement dynamic scoping:
[Vars, NumDims] = packageVariables(params, {'n_nrecevier_mod_1703', 'Transpose__5247_0'}, {n_nrecevier_mod_1703, Transpose__5247_0}, [n_nrecevier_mod_1703NumDims Transpose__5247_0NumDims]);
% Call the top-level graph function:
[n_nrecevier_mod_1808, n_nrecevier_mod_1800, n_nrecevier_mod_1799, n_nrecevier_mod_1808NumDims, n_nrecevier_mod_1800NumDims, n_nrecevier_mod_1799NumDims, state] = Shape_To_ReshapeGraph1035(n_nrecevier_mod_1703, Transpose__5247_0, NumDims.n_nrecevier_mod_1703, NumDims.Transpose__5247_0, Vars, NumDims, Training, params.State);
% Postprocess the output data
[n_nrecevier_mod_1808, n_nrecevier_mod_1800, n_nrecevier_mod_1799] = postprocessOutput(n_nrecevier_mod_1808, n_nrecevier_mod_1800, n_nrecevier_mod_1799, outputDataPerms, anyDlarrayInputs, Training, varargin{:});
end

function [n_nrecevier_mod_1808, n_nrecevier_mod_1800, n_nrecevier_mod_1799, n_nrecevier_mod_1808NumDims1042, n_nrecevier_mod_1800NumDims1043, n_nrecevier_mod_1799NumDims1044, state] = Shape_To_ReshapeGraph1035(n_nrecevier_mod_1703, Transpose__5247_0, n_nrecevier_mod_1703NumDims1040, Transpose__5247_0NumDims1041, Vars, NumDims, Training, state)
% Function implementing the graph 'Shape_To_ReshapeGraph1035'
% Update Vars and NumDims from the graph's formal input parameters. Note that state variables are already in Vars.
Vars.n_nrecevier_mod_1703 = n_nrecevier_mod_1703;
NumDims.n_nrecevier_mod_1703 = n_nrecevier_mod_1703NumDims1040;
Vars.Transpose__5247_0 = Transpose__5247_0;
NumDims.Transpose__5247_0 = Transpose__5247_0NumDims1041;

% Execute the operators:
% Shape:
[Vars.Shape__5633_0, NumDims.Shape__5633_0] = onnxShape(Vars.n_nrecevier_mod_1703, NumDims.n_nrecevier_mod_1703, 0, NumDims.n_nrecevier_mod_1703+1);

% Gather:
[Vars.n_nrecevier_mod_1681, NumDims.n_nrecevier_mod_1681] = onnxGather(Vars.Shape__5633_0, Vars.Const__5379, 0, NumDims.Shape__5633_0, NumDims.Const__5379);

% Cast:
if islogical(Vars.n_nrecevier_mod_1681)
    Vars.n_nrecevier_mod_1681 = single(Vars.n_nrecevier_mod_1681);
end
Vars.n_nrecevier_mod_1682 = cast(int32(extractdata(Vars.n_nrecevier_mod_1681)), 'like', Vars.n_nrecevier_mod_1681);
NumDims.n_nrecevier_mod_1682 = NumDims.n_nrecevier_mod_1681;

% Slice:
[Indices, NumDims.n_nrecevier_mod_1701] = prepareSliceArgs(Vars.n_nrecevier_mod_1682, Vars.const__2055, Vars.const_ends__3053, Vars.const_axes__4255, '', NumDims.n_nrecevier_mod_1682);
Vars.n_nrecevier_mod_1701 = subsref(Vars.n_nrecevier_mod_1682, Indices);

% Squeeze:
[Vars.n_nrecevier_mod_1700, NumDims.n_nrecevier_mod_1700] = onnxSqueeze(Vars.n_nrecevier_mod_1701, Vars.const_axes__4255, NumDims.n_nrecevier_mod_1701);

% Div:
Vars.n_nrecevier_mod_1691 = fix(Vars.n_nrecevier_mod_1700 ./ Vars.n_nrecevier_mod_535);
NumDims.n_nrecevier_mod_1691 = max(NumDims.n_nrecevier_mod_1700, NumDims.n_nrecevier_mod_535);

% Unsqueeze:
[shape, NumDims.n_nrecevier_mod_1675] = prepareUnsqueezeArgs(Vars.n_nrecevier_mod_1691, Vars.const_axes__4255, NumDims.n_nrecevier_mod_1691);
Vars.n_nrecevier_mod_1675 = reshape(Vars.n_nrecevier_mod_1691, shape);

% Concat:
[Vars.n_nrecevier_mod_1674, NumDims.n_nrecevier_mod_1674] = onnxConcat(0, {Vars.const_fold_opt__5935, Vars.const_fold_opt__5935, Vars.const_fold_opt__5935, Vars.const_fold_opt__5904, Vars.n_nrecevier_mod_1675}, [NumDims.const_fold_opt__5935, NumDims.const_fold_opt__5935, NumDims.const_fold_opt__5935, NumDims.const_fold_opt__5904, NumDims.n_nrecevier_mod_1675]);

% Cast:
if islogical(Vars.n_nrecevier_mod_1674)
    Vars.n_nrecevier_mod_1674 = single(Vars.n_nrecevier_mod_1674);
end
Vars.n_nrecevier_mod_1677 = cast(int64(extractdata(Vars.n_nrecevier_mod_1674)), 'like', Vars.n_nrecevier_mod_1674);
NumDims.n_nrecevier_mod_1677 = NumDims.n_nrecevier_mod_1674;

% Reshape:
[shape, NumDims.n_nrecevier_mod_1676] = prepareReshapeArgs(Vars.n_nrecevier_mod_1673, Vars.n_nrecevier_mod_1677, NumDims.n_nrecevier_mod_1673, 0);
Vars.n_nrecevier_mod_1676 = reshape(Vars.n_nrecevier_mod_1673, shape{:});

% Reshape:
[shape, NumDims.n_nrecevier_mod_1672] = prepareReshapeArgs(Vars.n_nrecevier_mod_1671, Vars.n_nrecevier_mod_1677, NumDims.n_nrecevier_mod_1671, 0);
Vars.n_nrecevier_mod_1672 = reshape(Vars.n_nrecevier_mod_1671, shape{:});

% Slice:
[Indices, NumDims.n_nrecevier_mod_1699] = prepareSliceArgs(Vars.n_nrecevier_mod_1682, Vars.const_starts__1983, Vars.const__738, Vars.const_axes__4255, '', NumDims.n_nrecevier_mod_1682);
Vars.n_nrecevier_mod_1699 = subsref(Vars.n_nrecevier_mod_1682, Indices);

% Slice:
[Indices, NumDims.n_nrecevier_mod_1698] = prepareSliceArgs(Vars.n_nrecevier_mod_1682, Vars.const_starts__1988, Vars.const_starts__1983, Vars.const_axes__4255, '', NumDims.n_nrecevier_mod_1682);
Vars.n_nrecevier_mod_1698 = subsref(Vars.n_nrecevier_mod_1682, Indices);

% Slice:
[Indices, NumDims.n_nrecevier_mod_1697] = prepareSliceArgs(Vars.n_nrecevier_mod_1682, Vars.const_axes__4255, Vars.const_starts__1988, Vars.const_axes__4255, '', NumDims.n_nrecevier_mod_1682);
Vars.n_nrecevier_mod_1697 = subsref(Vars.n_nrecevier_mod_1682, Indices);

% Concat:
[Vars.n_nrecevier_mod_1696, NumDims.n_nrecevier_mod_1696] = onnxConcat(0, {Vars.n_nrecevier_mod_1697, Vars.n_nrecevier_mod_1698, Vars.n_nrecevier_mod_1699, Vars.const_fold_opt__5904, Vars.n_nrecevier_mod_1675}, [NumDims.n_nrecevier_mod_1697, NumDims.n_nrecevier_mod_1698, NumDims.n_nrecevier_mod_1699, NumDims.const_fold_opt__5904, NumDims.n_nrecevier_mod_1675]);

% Cast:
if islogical(Vars.n_nrecevier_mod_1696)
    Vars.n_nrecevier_mod_1696 = single(Vars.n_nrecevier_mod_1696);
end
Vars.n_nrecevier_mod_1680 = cast(int64(extractdata(Vars.n_nrecevier_mod_1696)), 'like', Vars.n_nrecevier_mod_1696);
NumDims.n_nrecevier_mod_1680 = NumDims.n_nrecevier_mod_1696;

% Reshape:
[shape, NumDims.n_nrecevier_mod_1670] = prepareReshapeArgs(Vars.Transpose__5247_0, Vars.n_nrecevier_mod_1680, NumDims.Transpose__5247_0, 0);
Vars.n_nrecevier_mod_1670 = reshape(Vars.Transpose__5247_0, shape{:});

% ReduceMean:
dims = prepareReduceArgs(Vars.const_fold_opt__5883, NumDims.n_nrecevier_mod_1670);
Vars.n_nrecevier_mod_1694 = mean(Vars.n_nrecevier_mod_1670, dims);
NumDims.n_nrecevier_mod_1694 = NumDims.n_nrecevier_mod_1670;

% Sub:
Vars.n_nrecevier_mod_1692 = Vars.n_nrecevier_mod_1670 - Vars.n_nrecevier_mod_1694;
NumDims.n_nrecevier_mod_1692 = max(NumDims.n_nrecevier_mod_1670, NumDims.n_nrecevier_mod_1694);

% Mul:
Vars.n_nrecevier_mod_1693 = Vars.n_nrecevier_mod_1692 .* Vars.n_nrecevier_mod_1692;
NumDims.n_nrecevier_mod_1693 = max(NumDims.n_nrecevier_mod_1692, NumDims.n_nrecevier_mod_1692);

% ReduceMean:
dims = prepareReduceArgs(Vars.const_fold_opt__5883, NumDims.n_nrecevier_mod_1693);
Vars.n_nrecevier_mod_1695 = mean(Vars.n_nrecevier_mod_1693, dims);
NumDims.n_nrecevier_mod_1695 = NumDims.n_nrecevier_mod_1693;

% Add:
Vars.n_nrecevier_mod_1685 = Vars.n_nrecevier_mod_1695 + Vars.n_nrecevier_mod_146;
NumDims.n_nrecevier_mod_1685 = max(NumDims.n_nrecevier_mod_1695, NumDims.n_nrecevier_mod_146);

% Sqrt:
Vars.n_nrecevier_mod_1683 = sqrt(Vars.n_nrecevier_mod_1685);
NumDims.n_nrecevier_mod_1683 = NumDims.n_nrecevier_mod_1685;

% Reciprocal:
Vars.n_nrecevier_mod_1684 = 1./(Vars.n_nrecevier_mod_1683);
NumDims.n_nrecevier_mod_1684 = NumDims.n_nrecevier_mod_1683;

% Mul:
Vars.n_nrecevier_mod_1687 = Vars.n_nrecevier_mod_1684 .* Vars.n_nrecevier_mod_1672;
NumDims.n_nrecevier_mod_1687 = max(NumDims.n_nrecevier_mod_1684, NumDims.n_nrecevier_mod_1672);

% Mul:
Vars.n_nrecevier_mod_1689 = Vars.n_nrecevier_mod_1694 .* Vars.n_nrecevier_mod_1687;
NumDims.n_nrecevier_mod_1689 = max(NumDims.n_nrecevier_mod_1694, NumDims.n_nrecevier_mod_1687);

% Sub:
Vars.n_nrecevier_mod_1690 = Vars.n_nrecevier_mod_1676 - Vars.n_nrecevier_mod_1689;
NumDims.n_nrecevier_mod_1690 = max(NumDims.n_nrecevier_mod_1676, NumDims.n_nrecevier_mod_1689);

% Mul:
Vars.n_nrecevier_mod_1688 = Vars.n_nrecevier_mod_1670 .* Vars.n_nrecevier_mod_1687;
NumDims.n_nrecevier_mod_1688 = max(NumDims.n_nrecevier_mod_1670, NumDims.n_nrecevier_mod_1687);

% Add:
Vars.n_nrecevier_mod_1686 = Vars.n_nrecevier_mod_1688 + Vars.n_nrecevier_mod_1690;
NumDims.n_nrecevier_mod_1686 = max(NumDims.n_nrecevier_mod_1688, NumDims.n_nrecevier_mod_1690);

% Cast:
if islogical(Vars.n_nrecevier_mod_1682)
    Vars.n_nrecevier_mod_1682 = single(Vars.n_nrecevier_mod_1682);
end
Vars.n_nrecevier_mod_1679 = cast(int64(extractdata(Vars.n_nrecevier_mod_1682)), 'like', Vars.n_nrecevier_mod_1682);
NumDims.n_nrecevier_mod_1679 = NumDims.n_nrecevier_mod_1682;

% Reshape:
[shape, NumDims.n_nrecevier_mod_1678] = prepareReshapeArgs(Vars.n_nrecevier_mod_1686, Vars.n_nrecevier_mod_1679, NumDims.n_nrecevier_mod_1686, 0);
Vars.n_nrecevier_mod_1678 = reshape(Vars.n_nrecevier_mod_1686, shape{:});

% Relu:
Vars.n_nrecevier_mod_1636 = relu(Vars.n_nrecevier_mod_1678);
NumDims.n_nrecevier_mod_1636 = NumDims.n_nrecevier_mod_1678;

% Shape:
[Vars.n_nrecevier_mod_1809, NumDims.n_nrecevier_mod_1809] = onnxShape(Vars.n_nrecevier_mod_1636, NumDims.n_nrecevier_mod_1636, 0, NumDims.n_nrecevier_mod_1636+1);

% PLACEHOLDER FUNCTION FOR UNSUPPORTED OPERATOR (Size):
[Vars.n_nrecevier_mod_1825, NumDims.n_nrecevier_mod_1825] = PLACEHOLDER(Vars.n_nrecevier_mod_1809);

% Shape:
[Vars.n_nrecevier_mod_1806, NumDims.n_nrecevier_mod_1806] = onnxShape(Vars.n_nrecevier_mod_1636, NumDims.n_nrecevier_mod_1636, 0, NumDims.n_nrecevier_mod_1636+1);

% Cast:
if islogical(Vars.n_nrecevier_mod_1806)
    Vars.n_nrecevier_mod_1806 = single(Vars.n_nrecevier_mod_1806);
end
Vars.n_nrecevier_mod_1807 = cast(int32(extractdata(Vars.n_nrecevier_mod_1806)), 'like', Vars.n_nrecevier_mod_1806);
NumDims.n_nrecevier_mod_1807 = NumDims.n_nrecevier_mod_1806;

% Slice:
[Indices, NumDims.n_nrecevier_mod_1857] = prepareSliceArgs(Vars.n_nrecevier_mod_1807, Vars.const_starts__1983, Vars.const__738, Vars.const_axes__4255, '', NumDims.n_nrecevier_mod_1807);
Vars.n_nrecevier_mod_1857 = subsref(Vars.n_nrecevier_mod_1807, Indices);

% Slice:
[Indices, NumDims.n_nrecevier_mod_1856] = prepareSliceArgs(Vars.n_nrecevier_mod_1807, Vars.const_starts__1988, Vars.const_starts__1983, Vars.const_axes__4255, '', NumDims.n_nrecevier_mod_1807);
Vars.n_nrecevier_mod_1856 = subsref(Vars.n_nrecevier_mod_1807, Indices);

% Concat:
[Vars.n_nrecevier_mod_1855, NumDims.n_nrecevier_mod_1855] = onnxConcat(0, {Vars.n_nrecevier_mod_1856, Vars.n_nrecevier_mod_1857}, [NumDims.n_nrecevier_mod_1856, NumDims.n_nrecevier_mod_1857]);

% Add:
Vars.n_nrecevier_mod_1836 = Vars.n_nrecevier_mod_1855 + Vars.n_nrecevier_mod_938;
NumDims.n_nrecevier_mod_1836 = max(NumDims.n_nrecevier_mod_1855, NumDims.n_nrecevier_mod_938);

% Div:
Vars.Div__4222_0 = fix(Vars.n_nrecevier_mod_1836 ./ Vars.n_nrecevier_mod_937);
NumDims.Div__4222_0 = max(NumDims.n_nrecevier_mod_1836, NumDims.n_nrecevier_mod_937);

% Mul:
Vars.Mul__4223_0 = Vars.Div__4222_0 .* Vars.n_nrecevier_mod_937;
NumDims.Mul__4223_0 = max(NumDims.Div__4222_0, NumDims.n_nrecevier_mod_937);

% Sub:
Vars.n_nrecevier_mod_1843 = Vars.n_nrecevier_mod_1836 - Vars.Mul__4223_0;
NumDims.n_nrecevier_mod_1843 = max(NumDims.n_nrecevier_mod_1836, NumDims.Mul__4223_0);

% Sub:
Vars.n_nrecevier_mod_1854 = Vars.n_nrecevier_mod_937 - Vars.n_nrecevier_mod_1843;
NumDims.n_nrecevier_mod_1854 = max(NumDims.n_nrecevier_mod_937, NumDims.n_nrecevier_mod_1843);

% Div:
Vars.Div__4224_0 = fix(Vars.n_nrecevier_mod_1854 ./ Vars.n_nrecevier_mod_937);
NumDims.Div__4224_0 = max(NumDims.n_nrecevier_mod_1854, NumDims.n_nrecevier_mod_937);

% Mul:
Vars.Mul__4225_0 = Vars.Div__4224_0 .* Vars.n_nrecevier_mod_937;
NumDims.Mul__4225_0 = max(NumDims.Div__4224_0, NumDims.n_nrecevier_mod_937);

% Sub:
Vars.n_nrecevier_mod_1844 = Vars.n_nrecevier_mod_1854 - Vars.Mul__4225_0;
NumDims.n_nrecevier_mod_1844 = max(NumDims.n_nrecevier_mod_1854, NumDims.Mul__4225_0);

% Slice:
[Indices, NumDims.n_nrecevier_mod_1853] = prepareSliceArgs(Vars.n_nrecevier_mod_1844, Vars.const_starts__1988, Vars.const_starts__1983, Vars.const_axes__4255, '', NumDims.n_nrecevier_mod_1844);
Vars.n_nrecevier_mod_1853 = subsref(Vars.n_nrecevier_mod_1844, Indices);

% Concat:
[Vars.n_nrecevier_mod_1839, NumDims.n_nrecevier_mod_1839] = onnxConcat(0, {Vars.const_fold_opt__5837, Vars.n_nrecevier_mod_1853}, [NumDims.const_fold_opt__5837, NumDims.n_nrecevier_mod_1853]);

% Unsqueeze:
[shape, NumDims.n_nrecevier_mod_1842] = prepareUnsqueezeArgs(Vars.n_nrecevier_mod_1839, Vars.const_axes__4255, NumDims.n_nrecevier_mod_1839);
Vars.n_nrecevier_mod_1842 = reshape(Vars.n_nrecevier_mod_1839, shape);

% Slice:
[Indices, NumDims.n_nrecevier_mod_1852] = prepareSliceArgs(Vars.n_nrecevier_mod_1844, Vars.const_axes__4255, Vars.const_starts__1988, Vars.const_axes__4255, '', NumDims.n_nrecevier_mod_1844);
Vars.n_nrecevier_mod_1852 = subsref(Vars.n_nrecevier_mod_1844, Indices);

% Concat:
[Vars.n_nrecevier_mod_1838, NumDims.n_nrecevier_mod_1838] = onnxConcat(0, {Vars.const_fold_opt__5837, Vars.n_nrecevier_mod_1852}, [NumDims.const_fold_opt__5837, NumDims.n_nrecevier_mod_1852]);

% Unsqueeze:
[shape, NumDims.n_nrecevier_mod_1841] = prepareUnsqueezeArgs(Vars.n_nrecevier_mod_1838, Vars.const_axes__4255, NumDims.n_nrecevier_mod_1838);
Vars.n_nrecevier_mod_1841 = reshape(Vars.n_nrecevier_mod_1838, shape);

% Concat:
[Vars.n_nrecevier_mod_1840, NumDims.n_nrecevier_mod_1840] = onnxConcat(0, {Vars.n_nrecevier_mod_1841, Vars.n_nrecevier_mod_1842}, [NumDims.n_nrecevier_mod_1841, NumDims.n_nrecevier_mod_1842]);

% Cast:
if islogical(Vars.n_nrecevier_mod_1840)
    Vars.n_nrecevier_mod_1840 = single(Vars.n_nrecevier_mod_1840);
end
Vars.n_nrecevier_mod_1785 = cast(int64(extractdata(Vars.n_nrecevier_mod_1840)), 'like', Vars.n_nrecevier_mod_1840);
NumDims.n_nrecevier_mod_1785 = NumDims.n_nrecevier_mod_1840;

% Transpose:
[perm, NumDims.n_nrecevier_mod_1804] = prepareTransposeArgs('', NumDims.n_nrecevier_mod_1785);
if ~isempty(perm)
    Vars.n_nrecevier_mod_1804 = permute(Vars.n_nrecevier_mod_1785, perm);
end

% Slice:
[Indices, NumDims.n_nrecevier_mod_1798] = prepareSliceArgs(Vars.n_nrecevier_mod_1804, Vars.const__1051, Vars.const__1888, '', '', NumDims.n_nrecevier_mod_1804);
Vars.n_nrecevier_mod_1798 = subsref(Vars.n_nrecevier_mod_1804, Indices);

% Squeeze:
[Vars.n_nrecevier_mod_1800, NumDims.n_nrecevier_mod_1800] = onnxSqueeze(Vars.n_nrecevier_mod_1798, Vars.const_axes__4255, NumDims.n_nrecevier_mod_1798);

% Slice:
[Indices, NumDims.n_nrecevier_mod_1797] = prepareSliceArgs(Vars.n_nrecevier_mod_1804, Vars.const__773, Vars.const__774, '', '', NumDims.n_nrecevier_mod_1804);
Vars.n_nrecevier_mod_1797 = subsref(Vars.n_nrecevier_mod_1804, Indices);

% Squeeze:
[Vars.n_nrecevier_mod_1799, NumDims.n_nrecevier_mod_1799] = onnxSqueeze(Vars.n_nrecevier_mod_1797, Vars.const_axes__4255, NumDims.n_nrecevier_mod_1797);

% Add:
Vars.n_nrecevier_mod_1837 = Vars.n_nrecevier_mod_937 + Vars.n_nrecevier_mod_1844;
NumDims.n_nrecevier_mod_1837 = max(NumDims.n_nrecevier_mod_937, NumDims.n_nrecevier_mod_1844);

% Slice:
[Indices, NumDims.n_nrecevier_mod_1851] = prepareSliceArgs(Vars.n_nrecevier_mod_1837, Vars.const_starts__1988, Vars.const_starts__1983, Vars.const_axes__4255, '', NumDims.n_nrecevier_mod_1837);
Vars.n_nrecevier_mod_1851 = subsref(Vars.n_nrecevier_mod_1837, Indices);

% Concat:
[Vars.n_nrecevier_mod_1846, NumDims.n_nrecevier_mod_1846] = onnxConcat(0, {Vars.const_fold_opt__5913, Vars.n_nrecevier_mod_1851}, [NumDims.const_fold_opt__5913, NumDims.n_nrecevier_mod_1851]);

% Unsqueeze:
[shape, NumDims.n_nrecevier_mod_1849] = prepareUnsqueezeArgs(Vars.n_nrecevier_mod_1846, Vars.const_axes__4255, NumDims.n_nrecevier_mod_1846);
Vars.n_nrecevier_mod_1849 = reshape(Vars.n_nrecevier_mod_1846, shape);

% Slice:
[Indices, NumDims.n_nrecevier_mod_1850] = prepareSliceArgs(Vars.n_nrecevier_mod_1837, Vars.const_axes__4255, Vars.const_starts__1988, Vars.const_axes__4255, '', NumDims.n_nrecevier_mod_1837);
Vars.n_nrecevier_mod_1850 = subsref(Vars.n_nrecevier_mod_1837, Indices);

% Concat:
[Vars.n_nrecevier_mod_1845, NumDims.n_nrecevier_mod_1845] = onnxConcat(0, {Vars.const_fold_opt__5912, Vars.n_nrecevier_mod_1850}, [NumDims.const_fold_opt__5912, NumDims.n_nrecevier_mod_1850]);

% Unsqueeze:
[shape, NumDims.n_nrecevier_mod_1848] = prepareUnsqueezeArgs(Vars.n_nrecevier_mod_1845, Vars.const_axes__4255, NumDims.n_nrecevier_mod_1845);
Vars.n_nrecevier_mod_1848 = reshape(Vars.n_nrecevier_mod_1845, shape);

% Concat:
[Vars.n_nrecevier_mod_1847, NumDims.n_nrecevier_mod_1847] = onnxConcat(0, {Vars.n_nrecevier_mod_1848, Vars.n_nrecevier_mod_1849}, [NumDims.n_nrecevier_mod_1848, NumDims.n_nrecevier_mod_1849]);

% Cast:
if islogical(Vars.n_nrecevier_mod_1847)
    Vars.n_nrecevier_mod_1847 = single(Vars.n_nrecevier_mod_1847);
end
Vars.n_nrecevier_mod_1810 = cast(int64(extractdata(Vars.n_nrecevier_mod_1847)), 'like', Vars.n_nrecevier_mod_1847);
NumDims.n_nrecevier_mod_1810 = NumDims.n_nrecevier_mod_1847;

% Shape:
[Vars.n_nrecevier_mod_1823, NumDims.n_nrecevier_mod_1823] = onnxShape(Vars.n_nrecevier_mod_1810, NumDims.n_nrecevier_mod_1810, 0, NumDims.n_nrecevier_mod_1810+1);

% Slice:
[Indices, NumDims.n_nrecevier_mod_1826] = prepareSliceArgs(Vars.n_nrecevier_mod_1823, Vars.const_axes__4255, Vars.const_starts__1988, '', '', NumDims.n_nrecevier_mod_1823);
Vars.n_nrecevier_mod_1826 = subsref(Vars.n_nrecevier_mod_1823, Indices);

% Sub:
Vars.n_nrecevier_mod_1829 = Vars.n_nrecevier_mod_1825 - Vars.n_nrecevier_mod_1826;
NumDims.n_nrecevier_mod_1829 = max(NumDims.n_nrecevier_mod_1825, NumDims.n_nrecevier_mod_1826);

% Sub:
Vars.n_nrecevier_mod_1830 = Vars.n_nrecevier_mod_1829 - Vars.const_starts__1988;
NumDims.n_nrecevier_mod_1830 = max(NumDims.n_nrecevier_mod_1829, NumDims.const_starts__1988);

% Mul:
Vars.n_nrecevier_mod_1815 = Vars.const__1231 .* Vars.n_nrecevier_mod_1830;
NumDims.n_nrecevier_mod_1815 = max(NumDims.const__1231, NumDims.n_nrecevier_mod_1830);

% Pad:
[Vars.n_nrecevier_mod_1816, NumDims.n_nrecevier_mod_1816] = onnxPad(Vars.n_nrecevier_mod_1810, Vars.const__1230, 0, 'constant', dlarray([0:NumDims.n_nrecevier_mod_1810]'), NumDims.n_nrecevier_mod_1810);

% Pad:
[Vars.n_nrecevier_mod_1817, NumDims.n_nrecevier_mod_1817] = onnxPad(Vars.n_nrecevier_mod_1816, Vars.n_nrecevier_mod_1815, 0, 'constant', dlarray([0:NumDims.n_nrecevier_mod_1816]'), NumDims.n_nrecevier_mod_1816);

% Transpose:
[perm, NumDims.n_nrecevier_mod_1831] = prepareTransposeArgs('', NumDims.n_nrecevier_mod_1817);
if ~isempty(perm)
    Vars.n_nrecevier_mod_1831 = permute(Vars.n_nrecevier_mod_1817, perm);
end

% Reshape:
[shape, NumDims.n_nrecevier_mod_1819] = prepareReshapeArgs(Vars.n_nrecevier_mod_1831, Vars.const__2055, NumDims.n_nrecevier_mod_1831, 0);
Vars.n_nrecevier_mod_1819 = reshape(Vars.n_nrecevier_mod_1831, shape{:});

% Pad:
[Vars.n_nrecevier_mod_1818, NumDims.n_nrecevier_mod_1818] = onnxPad(Vars.n_nrecevier_mod_1636, Vars.n_nrecevier_mod_1819, 0, 'constant', dlarray([0:NumDims.n_nrecevier_mod_1636]'), NumDims.n_nrecevier_mod_1636);

% Shape:
[Vars.n_nrecevier_mod_1824, NumDims.n_nrecevier_mod_1824] = onnxShape(Vars.n_nrecevier_mod_1818, NumDims.n_nrecevier_mod_1818, 0, NumDims.n_nrecevier_mod_1818+1);

% Slice:
[Indices, NumDims.n_nrecevier_mod_1828] = prepareSliceArgs(Vars.n_nrecevier_mod_1824, Vars.const__738, Vars.const__664, '', '', NumDims.n_nrecevier_mod_1824);
Vars.n_nrecevier_mod_1828 = subsref(Vars.n_nrecevier_mod_1824, Indices);

% Slice:
[Indices, NumDims.n_nrecevier_mod_1827] = prepareSliceArgs(Vars.n_nrecevier_mod_1824, Vars.const_starts__1988, Vars.const__738, '', '', NumDims.n_nrecevier_mod_1824);
Vars.n_nrecevier_mod_1827 = subsref(Vars.n_nrecevier_mod_1824, Indices);

% Div:
Vars.n_nrecevier_mod_1814 = fix(Vars.n_nrecevier_mod_1827 ./ Vars.const__739);
NumDims.n_nrecevier_mod_1814 = max(NumDims.n_nrecevier_mod_1827, NumDims.const__739);

% Concat:
[Vars.n_nrecevier_mod_1813, NumDims.n_nrecevier_mod_1813] = onnxConcat(0, {Vars.const__2055, Vars.n_nrecevier_mod_1814, Vars.n_nrecevier_mod_1828}, [NumDims.const__2055, NumDims.n_nrecevier_mod_1814, NumDims.n_nrecevier_mod_1828]);

% Concat:
[Vars.n_nrecevier_mod_1811, NumDims.n_nrecevier_mod_1811] = onnxConcat(0, {Vars.n_nrecevier_mod_1814, Vars.const__739}, [NumDims.n_nrecevier_mod_1814, NumDims.const__739]);

% Reshape:
[shape, NumDims.n_nrecevier_mod_1820] = prepareReshapeArgs(Vars.n_nrecevier_mod_1811, Vars.const__1228, NumDims.n_nrecevier_mod_1811, 0);
Vars.n_nrecevier_mod_1820 = reshape(Vars.n_nrecevier_mod_1811, shape{:});

% Transpose:
[perm, NumDims.n_nrecevier_mod_1832] = prepareTransposeArgs('', NumDims.n_nrecevier_mod_1820);
if ~isempty(perm)
    Vars.n_nrecevier_mod_1832 = permute(Vars.n_nrecevier_mod_1820, perm);
end

% Reshape:
[shape, NumDims.n_nrecevier_mod_1821] = prepareReshapeArgs(Vars.n_nrecevier_mod_1832, Vars.const__2055, NumDims.n_nrecevier_mod_1832, 0);
Vars.n_nrecevier_mod_1821 = reshape(Vars.n_nrecevier_mod_1832, shape{:});

% Concat:
[Vars.n_nrecevier_mod_1812, NumDims.n_nrecevier_mod_1812] = onnxConcat(0, {Vars.const__2055, Vars.n_nrecevier_mod_1821, Vars.n_nrecevier_mod_1828}, [NumDims.const__2055, NumDims.n_nrecevier_mod_1821, NumDims.n_nrecevier_mod_1828]);

% Reshape:
[shape, NumDims.n_nrecevier_mod_1822] = prepareReshapeArgs(Vars.n_nrecevier_mod_1818, Vars.n_nrecevier_mod_1812, NumDims.n_nrecevier_mod_1818, 0);
Vars.n_nrecevier_mod_1822 = reshape(Vars.n_nrecevier_mod_1818, shape{:});

% Transpose:
[perm, NumDims.n_nrecevier_mod_1833] = prepareTransposeArgs(Vars.TransposePerm1039, NumDims.n_nrecevier_mod_1822);
if ~isempty(perm)
    Vars.n_nrecevier_mod_1833 = permute(Vars.n_nrecevier_mod_1822, perm);
end

% Reshape:
[shape, NumDims.n_nrecevier_mod_1808] = prepareReshapeArgs(Vars.n_nrecevier_mod_1833, Vars.n_nrecevier_mod_1813, NumDims.n_nrecevier_mod_1833, 0);
Vars.n_nrecevier_mod_1808 = reshape(Vars.n_nrecevier_mod_1833, shape{:});

% Set graph output arguments from Vars and NumDims:
n_nrecevier_mod_1808 = Vars.n_nrecevier_mod_1808;
n_nrecevier_mod_1808NumDims1042 = NumDims.n_nrecevier_mod_1808;
n_nrecevier_mod_1800 = Vars.n_nrecevier_mod_1800;
n_nrecevier_mod_1800NumDims1043 = NumDims.n_nrecevier_mod_1800;
n_nrecevier_mod_1799 = Vars.n_nrecevier_mod_1799;
n_nrecevier_mod_1799NumDims1044 = NumDims.n_nrecevier_mod_1799;
% Set output state from Vars:
state = updateStruct(state, Vars);
end

function [inputDataPerms, outputDataPerms, Training] = parseInputs(n_nrecevier_mod_1703, Transpose__5247_0, numDataOutputs, params, varargin)
% Function to validate inputs to Shape_To_ReshapeFcn:
p = inputParser;
isValidArrayInput = @(x)isnumeric(x) || isstring(x);
isValidONNXParameters = @(x)isa(x, 'ONNXParameters');
addRequired(p, 'n_nrecevier_mod_1703', isValidArrayInput);
addRequired(p, 'Transpose__5247_0', isValidArrayInput);
addRequired(p, 'params', isValidONNXParameters);
addParameter(p, 'InputDataPermutation', 'auto');
addParameter(p, 'OutputDataPermutation', 'auto');
addParameter(p, 'Training', false);
parse(p, n_nrecevier_mod_1703, Transpose__5247_0, params, varargin{:});
inputDataPerms = p.Results.InputDataPermutation;
outputDataPerms = p.Results.OutputDataPermutation;
Training = p.Results.Training;
if isnumeric(inputDataPerms)
    inputDataPerms = {inputDataPerms};
end
if isstring(inputDataPerms) && isscalar(inputDataPerms) || ischar(inputDataPerms)
    inputDataPerms = repmat({inputDataPerms},1,2);
end
if isnumeric(outputDataPerms)
    outputDataPerms = {outputDataPerms};
end
if isstring(outputDataPerms) && isscalar(outputDataPerms) || ischar(outputDataPerms)
    outputDataPerms = repmat({outputDataPerms},1,numDataOutputs);
end
end

function [n_nrecevier_mod_1703, Transpose__5247_0, Training, outputDataPerms, anyDlarrayInputs] = preprocessInput(n_nrecevier_mod_1703, Transpose__5247_0, params, varargin)
% Parse input arguments
[inputDataPerms, outputDataPerms, Training] = parseInputs(n_nrecevier_mod_1703, Transpose__5247_0, 3, params, varargin{:});
anyDlarrayInputs = any(cellfun(@(x)isa(x, 'dlarray'), {n_nrecevier_mod_1703, Transpose__5247_0}));
% Make the input variables into unlabelled dlarrays:
n_nrecevier_mod_1703 = makeUnlabeledDlarray(n_nrecevier_mod_1703);
Transpose__5247_0 = makeUnlabeledDlarray(Transpose__5247_0);
% Permute inputs if requested:
n_nrecevier_mod_1703 = permuteInputVar(n_nrecevier_mod_1703, inputDataPerms{1}, 4);
Transpose__5247_0 = permuteInputVar(Transpose__5247_0, inputDataPerms{2}, 4);
end

function [n_nrecevier_mod_1808, n_nrecevier_mod_1800, n_nrecevier_mod_1799] = postprocessOutput(n_nrecevier_mod_1808, n_nrecevier_mod_1800, n_nrecevier_mod_1799, outputDataPerms, anyDlarrayInputs, Training, varargin)
% Set output type:
if ~anyDlarrayInputs && ~Training
    if isdlarray(n_nrecevier_mod_1808)
        n_nrecevier_mod_1808 = extractdata(n_nrecevier_mod_1808);
    end
    if isdlarray(n_nrecevier_mod_1800)
        n_nrecevier_mod_1800 = extractdata(n_nrecevier_mod_1800);
    end
    if isdlarray(n_nrecevier_mod_1799)
        n_nrecevier_mod_1799 = extractdata(n_nrecevier_mod_1799);
    end
end
% Permute outputs if requested:
n_nrecevier_mod_1808 = permuteOutputVar(n_nrecevier_mod_1808, outputDataPerms{1}, 4);
end


%% dlarray functions implementing ONNX operators:

function [Y, numDimsY] = onnxConcat(ONNXAxis, XCell, numDimsXArray)
% Concatentation that treats all empties the same. Necessary because
% dlarray.cat does not allow, for example, cat(1, 1x1, 1x0) because the
% second dimension sizes do not match.

% Copyright 2021 The MathWorks, Inc.

numDimsY = numDimsXArray(1);
XCell(cellfun(@isempty, XCell)) = [];
if isempty(XCell)
    Y = dlarray([]);
else
    if ONNXAxis<0
        ONNXAxis = ONNXAxis + numDimsY;
    end
    DLTAxis = numDimsY - ONNXAxis;
    Y = cat(DLTAxis, XCell{:});
end
end

function [Y, numDimsY] = onnxGather(X, ONNXIdx, ONNXAxis, numDimsX, numDimsIdx)
% Function implementing the ONNX Gather operator

% In ONNX, 'Gather' first indexes into dimension ONNXAxis of data, using
% the contents of ONNXIdx as the indices. Then, it reshapes the ONNXAxis
% into the shape of ONNXIdx.
%   Example 1:
% Suppose data has shape [2 3 4 5], ONNXIdx has shape [6 7], and axis=1.
% The result has shape [2 6 7 4 5].
%   Example 2:
% Suppose data has shape [2 3 4 5], ONNXIdx has shape [6], and axis=1.
% The result has shape [2 6 4 5].
%   Example 3:
% Suppose data has shape [2 3 4 5], ONNXIdx has shape [] (a scalar), and axis=1.
% The result has shape [2 4 5].
%
% Since we're using reverse indexing relative to ONNX, in this function
% data and ONNXIdx both have reversed dimension ordering.

% Copyright 2020-2021 The MathWorks, Inc.

numDimsY = numDimsIdx + (numDimsX - 1);
if isempty(X)
    Y = X;
    return;
end
% (1) First, do the subsref part of Gather
if ONNXAxis<0
    ONNXAxis = ONNXAxis + numDimsX;                                 % Axis can be negative. Convert it to its positive equivalent.
end
dltAxis = numDimsX - ONNXAxis;                                      % Convert axis to DLT. ONNXAxis is origin 0 and we index from the end
ONNXIdx(ONNXIdx<0) = ONNXIdx(ONNXIdx<0) + size(X, dltAxis);         % ONNXIdx can have negative components. Make them positive.
dltIdx  = extractdata(ONNXIdx) + 1;                                 % ONNXIdx is origin-0 in ONNX, so add 1 to get dltIdx
% Use subsref to index into data
Indices.subs = repmat({':'}, 1, numDimsX);
Indices.subs{dltAxis} = dltIdx(:);                                  % Index as a column to ensure the output is 1-D in the indexed dimension (for now).
Indices.type = '()';
Y = subsref(X, Indices);
% (2) Now do the reshaping part of Gather
shape = size(Y, 1:numDimsX);
if numDimsIdx == 0
    % Delete the indexed dimension
    shape(dltAxis) = [];
elseif numDimsIdx > 1
    % Reshape the indexed dimension into the shape of ONNXIdx
    shape = [shape(1:dltAxis-1) size(ONNXIdx, 1:numDimsIdx) shape(dltAxis+1:end)];
end
% Extend the shape to 2D so it's valid MATLAB
if numel(shape) < 2
    shape = [shape ones(1,2-numel(shape))];
end
Y = reshape(Y, shape);
end

function [Y, numDimsY] = onnxPad(X, pads, value, mode, ONNXAxis, numDimsX)
% Implements the ONNX Pad operator

% ONNX 'pads' is a vector: [x1_begin, x2_begin...x1_end, x2_end,...], with
% x1,x2, listed in FORWARD ONNX dimension ordering, because it is data
% within a dimension and so is not flipped. xi_begin is the number of
% pixels added at the beginning of axis `i` and xi_end, the number of
% pixels added at the end of axis `i`.  pads can be negative, in which case
% that number of pixels is removed.

% Copyright 2020-2024 The MathWorks, Inc.

pads = pads(:)';
numDimsY = numDimsX;
if ONNXAxis < 0
    ONNXAxis = ONNXAxis + numDimsX;
end
% Fill in pads to length 2*numDimsX if size(ONNXAxis,1) < numDimsX
if size(ONNXAxis,1) < numDimsX
    helpPads = dlarray(zeros(1,2*numDimsX));
    helpPads([ONNXAxis+1,ONNXAxis+numDimsX+1]) = pads;
    pads = helpPads;
end

if numDimsX==1
    % X is Nx1. Temporarily make it reverse-ONNX 2D (1xN), then transpose
    % the result back to 1D at the end.
    X = X';
    numDimsX = 2;
    pads = [pads(1) 0 pads(2) 0];  % Don't pad the dummy dimension
    numDimsY = 1;
end
sizeX  = size(X, 1:numDimsX);
fwdPadMat = reshape(extractdata(pads), [], 2)';  % row1 = begins, row2 = ends
% Columns of padmat are in reverse ONNX ordering. Still the case that row1
% = begins, row2 = ends:
padmat = fliplr(fwdPadMat);
sizeY  = sum([sizeX; padmat]);
% Create output tensor of the right size
Y = value*ones(sizeY, 'like', X);
% Construct subsref indices for inserting (and cropping) the original
for i=1:numel(sizeX)
    Ysubs{i} = max(1,1+padmat(1,i)) : min(sizeY(i), sizeY(i)-padmat(2,i));
    Xsubs{i} = max(1,1-padmat(1,i)) : min(sizeX(i), sizeX(i)+padmat(2,i));
end
Sy      = struct('type', '()');
Sy.subs = Ysubs;
Sx      = struct('type', '()');
Sx.subs = Xsubs;
% Insert/crop the original into the result
Y = subsasgn(Y, Sy, subsref(X, Sx));
% Handle 'reflect' and 'edge' modes, but don't do it if X was 1D, 0x1.
if ismember(mode, ["edge", "reflect"]) && ~(numDimsY==1 && sizeX(2)==0)
    for dim = 1:numDimsX
        if any(padmat(:,dim)>0)
            % Setup a call to subsasgn
            prepad  = padmat(1,dim);
            postpad = padmat(2,dim);
            if prepad > 0
                [Sy, Sx] = prepadIndices(sizeX, prepad, dim, mode);
                Y = subsasgn(Y, Sy, subsref(Y, Sx));
            end
            if postpad > 0
                [Sy, Sx] = postpadIndices(sizeX, sizeY, prepad, postpad, dim, mode);
                Y = subsasgn(Y, Sy, subsref(Y, Sx));
            end
        end
    end
end
% Transpose the result back to 1D if the input was 1D
if numDimsY==1
    Y = Y';
end

% Subfunctions in onnxPad:
    function [Sy, Sx] = prepadIndices(sizeX, prepad, dim, mode)
        Sy   	= struct('type', '()');
        Sy.subs	= repmat({':'}, [1 numel(sizeX)]);
        Sx   	= Sy;
        % Write into the first 'prepad' elements of Y.dim.
        Sy.subs{dim} = 1:prepad;
        switch mode
            case 'reflect'
                % Create indices 2:prepad+1 of X.dim, in the reverse order, with
                % wraparound. Then add prepad to convert them to Y indices.
                Sx.subs{dim} = wrapIndices(prepad+1 : -1 : 2, sizeX(dim)) + prepad;
            case 'edge'
                % Create replicated indices 1 of X.dim. Then add prepad to
                % convert them to Y indices.
                Sx.subs{dim} = repmat(1, [1 prepad]) + prepad;
            otherwise
                assert(false);
        end
    end

    function [Sy, Sx] = postpadIndices(sizeX, sizeY, prepad, postpad, dim, mode)
        Sy   	= struct('type', '()');
        Sy.subs	= repmat({':'}, [1 numel(sizeX)]);
        Sx   	= Sy;
        % Write into the last 'postpad' elements of Y.dim.
        Sy.subs{dim} = sizeY(dim)-postpad+1 : sizeY(dim);
        switch mode
            case 'reflect'
                % Create indices in the reverse order, with wraparound. Then add
                % prepad to convert them to Y indices.
                Sx.subs{dim} = wrapIndices(sizeX(dim)-1 : -1 : sizeX(dim)-postpad, sizeX(dim)) + prepad;
            case 'edge'
                % Create replicated end indices . Then add prepad to convert them
                % to Y indices.
                Sx.subs{dim} = repmat(sizeX(dim), [1 postpad]) + prepad;
            otherwise
                assert(false);
        end
    end

    function j = wrapIndices(i, maxIdx)
        % i can be positive, negative or zero. Legal output indices are in the
        % range 1:maxIdx.
        j = mod(i-1, maxIdx) + 1;
    end
end


function [Y, numDimsY] = onnxShape(X, numDimsX, startAxis, endAxis)
% Implements the ONNX Shape operator
% Return the reverse ONNX shape as a 1D column vector

% Copyright 2020-2024 The MathWorks, Inc.

switch numDimsX
    case 0
        if isempty(X)
            Y = dlarray(0);
        else
            Y = dlarray(1);
        end
    case 1
        if isempty(X)
            Y = dlarray(0);
        else
            Y = dlarray(size(X,1));
        end
    otherwise
        if(endAxis<0)
            %  If the endAxis is smaller than 0 after converting it positive,
            % the endAxis is 0
            endAxis = max(0, numDimsX + endAxis);
        end
        if(startAxis<0)
            %  If the startAxis is smaller than 0 after converting it positive,
            % the startAxis is 0
            startAxis = max(0, numDimsX + startAxis);
        end
        % transform startAxis and endAxis from 0 index to 1 index
        startAxis = startAxis + 1;
        endAxis = endAxis + 1;
        % if startAxis is larger than numDimsX or endAxis is larger than
        % numDimsX + 1, cramp it to the upper bound. The endAxis is exclusive,
        % transform it to MATLAB inclusive way
        endAxis = min(endAxis, numDimsX + 1) - 1;
        startAxis = min(startAxis, numDimsX);
        if endAxis < startAxis || endAxis == 0
            Y = dlarray(0);
        else
            Y = dlarray(fliplr(size(X, (numDimsX-endAxis+1):(numDimsX-startAxis+1)))');
        end
end
numDimsY = 1;
end

function [Y, numDimsY] = onnxSqueeze(X, ONNXAxes, numDimsX)
% Implements the ONNX Squeeze operator

% Copyright 2020 The MathWorks, Inc.

if numDimsX == 0
    Y = X;
    numDimsY = numDimsX;
else
    % Find the new ONNX shape
    curOShape = size(X, numDimsX:-1:1);
    if isempty(ONNXAxes)
        newOShape = curOShape(curOShape ~= 1);
    else
        ONNXAxes(ONNXAxes<0) = ONNXAxes(ONNXAxes<0) + numDimsX;
        newOShape = curOShape;
        newOShape(ONNXAxes+1) = [];
    end
    % Get numDimsY from ONNX shape
    numDimsY  = numel(newOShape);
    newMShape = [fliplr(newOShape) ones(1, 2-numDimsY)];    % Append 1's to shape if numDims<2
    Y         = reshape(X, newMShape);
end
end

function dims = prepareReduceArgs(ONNXAxes, numDimsX)
% Prepares arguments for implementing the ONNX Reduce operator

%   Copyright 2020 The MathWorks, Inc.

if isempty(ONNXAxes)
    ONNXAxes = 0:numDimsX-1;   % All axes
end
ONNXAxes(ONNXAxes<0) = ONNXAxes(ONNXAxes<0) + numDimsX;
dims = numDimsX - ONNXAxes;
end

function [DLTShape, numDimsY] = prepareReshapeArgs(X, ONNXShape, numDimsX, allowzero)
% Prepares arguments for implementing the ONNX Reshape operator

%   Copyright 2020-2024 The MathWorks, Inc.

ONNXShape = flip(extractdata(ONNXShape));            % First flip the shape to make it correspond to the dimensions of X.
% In ONNX, 0 means "unchanged" if allowzero is false, and -1 means "infer". In DLT, there is no
% "unchanged", and [] means "infer".
DLTShape = num2cell(ONNXShape);                      % Make a cell array so we can include [].
% Replace zeros with the actual size if allowzero is false
if any(ONNXShape==0) && allowzero==0
    i0 = find(ONNXShape==0);
    DLTShape(i0) = num2cell(size(X, numDimsX - numel(ONNXShape) + i0));  % right-align the shape vector and dims
end
if any(ONNXShape == -1)
    % Replace -1 with []
    i = ONNXShape == -1;
    DLTShape{i} = [];
end
if numel(DLTShape)==1
    DLTShape = [DLTShape 1];
end
numDimsY = numel(ONNXShape);
end

function [S, numDimsY] = prepareSliceArgs(X, Starts, Ends, Axes, Steps, numDimsX)
% Prepares arguments for implementing the ONNX Slice operator

%   Copyright 2020 The MathWorks, Inc.

% Starts, Ends and Axes are all origin 0. Axes refer to the ONNX dimension
% ordering, but X uses the reverse, DLT ordering. Starts, Ends, Axes, and
% Steps correspond positionally. Axes and Steps may be omitted, with
% defaults described in the ONNX spec.

% Set default Axes and Steps if not supplied
if isempty(Axes)
    Axes = 0:numDimsX-1;   % All axes
end
Axes(Axes<0) = Axes(Axes<0) + numDimsX; % Handle negative Axes.
if isempty(Steps)
    Steps = ones(1, numel(Starts));
end
% Init all dims to :
S.subs = repmat({':'}, 1, numDimsX);
S.type = '()';
% Set Starts and Ends for each axis
for i = 1:numel(Axes)
    DLTDim = numDimsX - Axes(i);                                               % The DLT dim is the reverse of the ONNX dim.
    % "If a negative value is passed for any of the start or end indices,
    % it represents number of elements before the end of that dimension."
    if Starts(i) < 0
        Starts(i) = size(X,DLTDim) + Starts(i);
    end
    if Ends(i) < 0
        Ends(i) = max(-1, size(X,DLTDim) + Ends(i));                        % The -1 case is when we're slicing backward and want to include 0.
    end
    % "If the value passed to start or end is larger than the n (the number
    % of elements in this dimension), it represents n."
    if Starts(i) > size(X,DLTDim)
        Starts(i) = size(X,DLTDim);
    end
    if Ends(i) > size(X,DLTDim)
        Ends(i) = size(X,DLTDim);
    end
    if Steps(i) > 0
        S.subs{DLTDim} = 1 + (Starts(i) : Steps(i) : Ends(i)-1);            % 1 + (Origin 0 indexing with end index excluded)
    else
        S.subs{DLTDim} = 1 + (Starts(i) : Steps(i) : Ends(i)+1);            % 1 + (Origin 0 indexing with end index excluded)
    end
end
numDimsY = numDimsX;
end

function [perm, numDimsA] = prepareTransposeArgs(ONNXPerm, numDimsA)
% Prepares arguments for implementing the ONNX Transpose operator

%   Copyright 2020 The MathWorks, Inc.

if numDimsA <= 1        % Tensors of numDims 0 or 1 are unchanged by ONNX Transpose.
    perm = [];
else
    if isempty(ONNXPerm)        % Empty ONNXPerm means reverse the dimensions.
        perm = numDimsA:-1:1;
    else
        perm = numDimsA-flip(ONNXPerm);
    end
end
end

function [newShape, numDimsY] = prepareUnsqueezeArgs(X, ONNXAxes, numDimsX)
% Prepares arguments for implementing the ONNX Unsqueeze operator

%   Copyright 2020-2021 The MathWorks, Inc.

numDimsY = numDimsX + numel(ONNXAxes);
ONNXAxes = extractdata(ONNXAxes);
ONNXAxes(ONNXAxes<0) = ONNXAxes(ONNXAxes<0) + numDimsY;
ONNXAxes = sort(ONNXAxes);                                              % increasing order
if numDimsY == 1
    newShape = size(X);
else
    DLTAxes  = flip(numDimsY - ONNXAxes);                                  % increasing order
    newShape = ones(1, numDimsY);
    posToSet = setdiff(1:numDimsY, DLTAxes, 'stable');
    newShape(posToSet) = size(X, 1:numel(posToSet));
end
end

%% Utility functions:

function s = appendStructs(varargin)
% s = appendStructs(s1, s2,...). Assign all fields in s1, s2,... into s.

%   Copyright 2020 The MathWorks, Inc.

if isempty(varargin)
    s = struct;
else
    s = varargin{1};
    for i = 2:numel(varargin)
        fromstr = varargin{i};
        fs = fieldnames(fromstr);
        for j = 1:numel(fs)
            s.(fs{j}) = fromstr.(fs{j});
        end
    end
end
end

function checkInputSize(inputShape, expectedShape, inputName)

%   Copyright 2020-2021 The MathWorks, Inc.

if numel(expectedShape)==0
    % The input is a scalar
    if ~isequal(inputShape, [1 1])
        inputSizeStr = makeSizeString(inputShape);
        error(message('nnet_cnn_onnx:onnx:InputNeedsResize',inputName, "[1,1]", inputSizeStr));
    end
elseif numel(expectedShape)==1
    % The input is a vector
    if ~shapeIsColumnVector(inputShape) || ~iSizesMatch({inputShape(1)}, expectedShape)
        expectedShape{2} = 1;
        expectedSizeStr = makeSizeString(expectedShape);
        inputSizeStr = makeSizeString(inputShape);
        error(message('nnet_cnn_onnx:onnx:InputNeedsResize',inputName, expectedSizeStr, inputSizeStr));
    end
else
    % The input has 2 dimensions or more

    % The input dimensions have been reversed; flip them back to compare to the
    % expected ONNX shape.
    inputShape = fliplr(inputShape);

    % If the expected shape has fewer dims than the input shape, error.
    if numel(expectedShape) < numel(inputShape)
        expectedSizeStr = strjoin(["[", strjoin(string(expectedShape), ","), "]"], "");
        error(message('nnet_cnn_onnx:onnx:InputHasGreaterNDims', inputName, expectedSizeStr));
    end

    % Prepad the input shape with trailing ones up to the number of elements in
    % expectedShape
    inputShape = num2cell([ones(1, numel(expectedShape) - length(inputShape)) inputShape]);

    % Find the number of variable size dimensions in the expected shape
    numVariableInputs = sum(cellfun(@(x) isa(x, 'char') || isa(x, 'string'), expectedShape));

    % Find the number of input dimensions that are not in the expected shape
    % and cannot be represented by a variable dimension
    nonMatchingInputDims = setdiff(string(inputShape), string(expectedShape));
    numNonMatchingInputDims  = numel(nonMatchingInputDims) - numVariableInputs;

    expectedSizeStr = makeSizeString(expectedShape);
    inputSizeStr = makeSizeString(inputShape);
    if numNonMatchingInputDims == 0 && ~iSizesMatch(inputShape, expectedShape)
        % The actual and expected input dimensions match, but in
        % a different order. The input needs to be permuted.
        error(message('nnet_cnn_onnx:onnx:InputNeedsPermute',inputName, expectedSizeStr, inputSizeStr));
    elseif numNonMatchingInputDims > 0
        % The actual and expected input sizes do not match.
        error(message('nnet_cnn_onnx:onnx:InputNeedsResize',inputName, expectedSizeStr, inputSizeStr));
    end
end
end

function doesMatch = iSizesMatch(inputShape, expectedShape)
% Check whether the input and expected shapes match, in order.
% Size elements match if (1) the elements are equal, or (2) the expected
% size element is a variable (represented by a character vector or string)
doesMatch = true;
for i=1:numel(inputShape)
    if ~(isequal(inputShape{i},expectedShape{i}) || ischar(expectedShape{i}) || isstring(expectedShape{i}))
        doesMatch = false;
        return
    end
end
end

function sizeStr = makeSizeString(shape)
sizeStr = strjoin(["[", strjoin(string(shape), ","), "]"], "");
end

function isVec = shapeIsColumnVector(shape)
if numel(shape) == 2 && shape(2) == 1
    isVec = true;
else
    isVec = false;
end
end
function X = makeUnlabeledDlarray(X)
% Make numeric X into an unlabelled dlarray

%   Copyright 2020-2021 The MathWorks, Inc.

if isa(X, 'dlarray')
    X = stripdims(X);
elseif isnumeric(X)
    if isinteger(X)
        % Make ints double so they can combine with anything without
        % reducing precision
        X = double(X);
    end
    X = dlarray(X);
end
end

function [Vars, NumDims] = packageVariables(params, inputNames, inputValues, inputNumDims)

%   Copyright 2020 The MathWorks, Inc.

% inputNames, inputValues are cell arrays. inputRanks is a numeric vector.
Vars = appendStructs(params.Learnables, params.Nonlearnables, params.State);
NumDims = params.NumDimensions;
% Add graph inputs
for i = 1:numel(inputNames)
    Vars.(inputNames{i}) = inputValues{i};
    NumDims.(inputNames{i}) = inputNumDims(i);
end
end

function X = permuteInputVar(X, userDataPerm, onnxNDims)

%   Copyright 2020-2021 The MathWorks, Inc.
% Returns reverse-ONNX ordering
if onnxNDims == 0
    return;
elseif onnxNDims == 1 && isvector(X)
    X = X(:);
    return;
elseif isnumeric(userDataPerm)
    % Permute into reverse ONNX ordering
    if numel(userDataPerm) ~= onnxNDims
        error(message('nnet_cnn_onnx:onnx:InputPermutationSize', numel(userDataPerm), onnxNDims));
    end
    perm = fliplr(userDataPerm);
elseif isequal(userDataPerm, 'auto') && onnxNDims == 4
    % Permute MATLAB HWCN to reverse onnx (WHCN)
    perm = [2 1 3 4];
elseif isequal(userDataPerm, 'as-is')
    % Do not permute the input
    perm = 1:ndims(X);
else
    % userDataPerm is either 'none' or 'auto' with no default, which means
    % it's already in onnx ordering, so just make it reverse onnx
    perm = max(2,onnxNDims):-1:1;
end
X = permute(X, perm);
end

function Y = permuteOutputVar(Y, userDataPerm, onnxNDims)

%   Copyright 2020-2021 The MathWorks, Inc.
switch onnxNDims
    case 0
        perm = [];
    case 1
        if isnumeric(userDataPerm)
            % Use the user's permutation because Y is a column vector which
            % already matches ONNX.
            perm = userDataPerm;
        elseif isequal(userDataPerm, 'auto')
            % Treat the 1D onnx vector as a 2D column and transpose it
            perm = [2 1];
        else
            % userDataPerm is 'none'. Leave Y alone because it already
            % matches onnx.
            perm = [];
        end
    otherwise
        % ndims >= 2
        if isnumeric(userDataPerm)
            % Use the inverse of the user's permutation. This is not just the
            % flip of the permutation vector.
            perm = onnxNDims + 1 - userDataPerm;
        elseif isequal(userDataPerm, 'auto')
            if onnxNDims == 2
                % Permute reverse ONNX CN to DLT CN (do nothing)
                perm = [];
            elseif onnxNDims == 4
                % Permute reverse onnx (WHCN) to MATLAB HWCN
                perm = [2 1 3 4];
            else
                % User wants the output in ONNX ordering, so just reverse it from
                % reverse onnx
                perm = onnxNDims:-1:1;
            end
        elseif isequal(userDataPerm, 'as-is')
            % Do not permute the input
            perm = 1:ndims(Y);
        else
            % userDataPerm is 'none', so just make it reverse onnx
            perm = onnxNDims:-1:1;
        end
end
if ~isempty(perm)
    Y = permute(Y, perm);
end
end

function s = updateStruct(s, t)
% Set all existing fields in s from fields in t, ignoring extra fields in
% t.
%   Copyright 2020 The MathWorks, Inc.

for name = transpose(fieldnames(s))
    s.(name{1}) = t.(name{1});
end
end
