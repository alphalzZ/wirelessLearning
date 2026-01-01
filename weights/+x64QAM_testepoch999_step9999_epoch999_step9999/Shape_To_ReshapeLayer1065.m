classdef Shape_To_ReshapeLayer1065 < nnet.layer.Layer & nnet.layer.Formattable
    % A custom layer auto-generated while importing an ONNX network.

    %#codegen
    %#ok<*PROPLC>
    %#ok<*NBRAK>
    %#ok<*INUSL>
    %#ok<*VARARG>

    properties (Learnable)
        const_fold_opt__5883
        n_nrecevier_mod_1639
        n_nrecevier_mod_1643
    end

    properties
        ONNXParams         % An ONNXParameters object containing parameters used by this layer.
    end

    methods
        function this = Shape_To_ReshapeLayer1065(name, onnxParams)
            this.Name = name;
            this.NumInputs = 2;
            this.NumOutputs = 5;
            this.OutputNames = {'n_nrecevier_mod_1730', 'n_nrecevier_mod_1722', 'n_nrecevier_mod_1721', 'n_nrecevier_mod_1722NumDims', 'n_nrecevier_mod_1721NumDims'};
            this.ONNXParams = onnxParams;
            this.const_fold_opt__5883 = onnxParams.Learnables.const_fold_opt__5883;
            this.n_nrecevier_mod_1639 = onnxParams.Learnables.n_nrecevier_mod_1639;
            this.n_nrecevier_mod_1643 = onnxParams.Learnables.n_nrecevier_mod_1643;
        end

        function [n_nrecevier_mod_1730, n_nrecevier_mod_1722, n_nrecevier_mod_1721, n_nrecevier_mod_1722NumDims, n_nrecevier_mod_1721NumDims] = predict(this, n_nrecevier_mod_1412, Transpose__5231_0)
            if isdlarray(n_nrecevier_mod_1412)
                n_nrecevier_mod_1412 = stripdims(n_nrecevier_mod_1412);
            end
            if isdlarray(Transpose__5231_0)
                Transpose__5231_0 = stripdims(Transpose__5231_0);
            end
            n_nrecevier_mod_1412NumDims = 4;
            Transpose__5231_0NumDims = 4;
            onnxParams = this.ONNXParams;
            onnxParams.Learnables.const_fold_opt__5883 = this.const_fold_opt__5883;
            onnxParams.Learnables.n_nrecevier_mod_1639 = this.n_nrecevier_mod_1639;
            onnxParams.Learnables.n_nrecevier_mod_1643 = this.n_nrecevier_mod_1643;
            [n_nrecevier_mod_1730, n_nrecevier_mod_1722, n_nrecevier_mod_1721, n_nrecevier_mod_1730NumDims, n_nrecevier_mod_1722NumDims, n_nrecevier_mod_1721NumDims] = Shape_To_ReshapeFcn(n_nrecevier_mod_1412, Transpose__5231_0, n_nrecevier_mod_1412NumDims, Transpose__5231_0NumDims, onnxParams, 'Training', false, ...
                'InputDataPermutation', {[4 3 1 2], [4 1 2 3], ['as-is'], ['as-is']}, ...
                'OutputDataPermutation', {[2 3 4 1], ['as-is'], ['as-is'], ['as-is'], ['as-is'], ['as-is']});
            if any(cellfun(@(A)~isnumeric(A) && ~islogical(A), {n_nrecevier_mod_1730, n_nrecevier_mod_1722, n_nrecevier_mod_1721}))
                fprintf('Runtime error in network. At least one output of custom layer ''%s'' is a non-numeric, non-logical value.\n', 'Shape_To_ReshapeLayer1065');
                error(message('nnet_cnn_onnx:onnx:BadCustomLayerRuntimeOutput', 'Shape_To_ReshapeLayer1065'));
            end
            n_nrecevier_mod_1730 = dlarray(single(n_nrecevier_mod_1730), 'SSCB');
            n_nrecevier_mod_1722 = dlarray(single(n_nrecevier_mod_1722), repmat('U', 1, max(2, n_nrecevier_mod_1722NumDims)));
            n_nrecevier_mod_1721 = dlarray(single(n_nrecevier_mod_1721), repmat('U', 1, max(2, n_nrecevier_mod_1721NumDims)));
            n_nrecevier_mod_1722NumDims = dlarray(ones(1,n_nrecevier_mod_1722NumDims,'like',n_nrecevier_mod_1730), 'UU');
            n_nrecevier_mod_1721NumDims = dlarray(ones(1,n_nrecevier_mod_1721NumDims,'like',n_nrecevier_mod_1730), 'UU');
            if ~coder.target('MATLAB')
                n_nrecevier_mod_1730 = extractdata(n_nrecevier_mod_1730);
                n_nrecevier_mod_1722 = extractdata(n_nrecevier_mod_1722);
                n_nrecevier_mod_1721 = extractdata(n_nrecevier_mod_1721);
                n_nrecevier_mod_1722NumDims = extractdata(n_nrecevier_mod_1722NumDims);
                n_nrecevier_mod_1721NumDims = extractdata(n_nrecevier_mod_1721NumDims);
            end
        end

        function [n_nrecevier_mod_1730, n_nrecevier_mod_1722, n_nrecevier_mod_1721, n_nrecevier_mod_1722NumDims, n_nrecevier_mod_1721NumDims] = forward(this, n_nrecevier_mod_1412, Transpose__5231_0)
            if isdlarray(n_nrecevier_mod_1412)
                n_nrecevier_mod_1412 = stripdims(n_nrecevier_mod_1412);
            end
            if isdlarray(Transpose__5231_0)
                Transpose__5231_0 = stripdims(Transpose__5231_0);
            end
            n_nrecevier_mod_1412NumDims = 4;
            Transpose__5231_0NumDims = 4;
            onnxParams = this.ONNXParams;
            onnxParams.Learnables.const_fold_opt__5883 = this.const_fold_opt__5883;
            onnxParams.Learnables.n_nrecevier_mod_1639 = this.n_nrecevier_mod_1639;
            onnxParams.Learnables.n_nrecevier_mod_1643 = this.n_nrecevier_mod_1643;
            [n_nrecevier_mod_1730, n_nrecevier_mod_1722, n_nrecevier_mod_1721, n_nrecevier_mod_1730NumDims, n_nrecevier_mod_1722NumDims, n_nrecevier_mod_1721NumDims] = Shape_To_ReshapeFcn(n_nrecevier_mod_1412, Transpose__5231_0, n_nrecevier_mod_1412NumDims, Transpose__5231_0NumDims, onnxParams, 'Training', true, ...
                'InputDataPermutation', {[4 3 1 2], [4 1 2 3], ['as-is'], ['as-is']}, ...
                'OutputDataPermutation', {[2 3 4 1], ['as-is'], ['as-is'], ['as-is'], ['as-is'], ['as-is']});
            if any(cellfun(@(A)~isnumeric(A) && ~islogical(A), {n_nrecevier_mod_1730, n_nrecevier_mod_1722, n_nrecevier_mod_1721}))
                fprintf('Runtime error in network. At least one output of custom layer ''%s'' is a non-numeric, non-logical value.\n', 'Shape_To_ReshapeLayer1065');
                error(message('nnet_cnn_onnx:onnx:BadCustomLayerRuntimeOutput', 'Shape_To_ReshapeLayer1065'));
            end
            n_nrecevier_mod_1730 = dlarray(single(n_nrecevier_mod_1730), 'SSCB');
            n_nrecevier_mod_1722 = dlarray(single(n_nrecevier_mod_1722), repmat('U', 1, max(2, n_nrecevier_mod_1722NumDims)));
            n_nrecevier_mod_1721 = dlarray(single(n_nrecevier_mod_1721), repmat('U', 1, max(2, n_nrecevier_mod_1721NumDims)));
            n_nrecevier_mod_1722NumDims = dlarray(ones(1,n_nrecevier_mod_1722NumDims,'like',n_nrecevier_mod_1730), 'UU');
            n_nrecevier_mod_1721NumDims = dlarray(ones(1,n_nrecevier_mod_1721NumDims,'like',n_nrecevier_mod_1730), 'UU');
            if ~coder.target('MATLAB')
                n_nrecevier_mod_1730 = extractdata(n_nrecevier_mod_1730);
                n_nrecevier_mod_1722 = extractdata(n_nrecevier_mod_1722);
                n_nrecevier_mod_1721 = extractdata(n_nrecevier_mod_1721);
                n_nrecevier_mod_1722NumDims = extractdata(n_nrecevier_mod_1722NumDims);
                n_nrecevier_mod_1721NumDims = extractdata(n_nrecevier_mod_1721NumDims);
            end
        end
    end
end

function [n_nrecevier_mod_1730, n_nrecevier_mod_1722, n_nrecevier_mod_1721, n_nrecevier_mod_1730NumDims, n_nrecevier_mod_1722NumDims, n_nrecevier_mod_1721NumDims, state] = Shape_To_ReshapeFcn(n_nrecevier_mod_1412, Transpose__5231_0, n_nrecevier_mod_1412NumDims, Transpose__5231_0NumDims, params, varargin)
%SHAPE_TO_RESHAPEFCN Function implementing an imported ONNX network.
%
% THIS FILE WAS AUTO-GENERATED BY importONNXFunction.
% ONNX Operator Set Version: 18
%
% Variable names in this function are taken from the original ONNX file.
%
% [N_NRECEVIER_MOD_1730, N_NRECEVIER_MOD_1722, N_NRECEVIER_MOD_1721] = Shape_To_ReshapeFcn(N_NRECEVIER_MOD_1412, TRANSPOSE__5231_0, PARAMS)
%			- Evaluates the imported ONNX network SHAPE_TO_RESHAPEFCN with input(s)
%			N_NRECEVIER_MOD_1412, TRANSPOSE__5231_0 and the imported network parameters in PARAMS. Returns
%			network output(s) in N_NRECEVIER_MOD_1730, N_NRECEVIER_MOD_1722, N_NRECEVIER_MOD_1721.
%
% [N_NRECEVIER_MOD_1730, N_NRECEVIER_MOD_1722, N_NRECEVIER_MOD_1721, STATE] = Shape_To_ReshapeFcn(N_NRECEVIER_MOD_1412, TRANSPOSE__5231_0, PARAMS)
%			- Additionally returns state variables in STATE. When training,
%			use this form and set TRAINING to true.
%
% [__] = Shape_To_ReshapeFcn(N_NRECEVIER_MOD_1412, TRANSPOSE__5231_0, PARAMS, 'NAME1', VAL1, 'NAME2', VAL2, ...)
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
% N_NRECEVIER_MOD_1412, TRANSPOSE__5231_0
%			- Input(s) to the ONNX network.
%			  The input size(s) expected by the ONNX file are:
%				  N_NRECEVIER_MOD_1412:		[Unknown, Unknown, Unknown, Unknown]				Type: FLOAT
%				  TRANSPOSE__5231_0:		[Unknown, Unknown, Unknown, Unknown]				Type: FLOAT
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
% N_NRECEVIER_MOD_1730, N_NRECEVIER_MOD_1722, N_NRECEVIER_MOD_1721
%			- Output(s) of the ONNX network.
%			  Without permutation, the size(s) of the outputs are:
%				  N_NRECEVIER_MOD_1730:		[Unknown, Unknown, Unknown, Unknown]				Type: FLOAT
%				  N_NRECEVIER_MOD_1722:		[1, 1]				Type: FLOAT
%				  N_NRECEVIER_MOD_1721:		[1, 1]				Type: FLOAT
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
[n_nrecevier_mod_1412, Transpose__5231_0, Training, outputDataPerms, anyDlarrayInputs] = preprocessInput(n_nrecevier_mod_1412, Transpose__5231_0, params, varargin{:});
% Put all variables into a single struct to implement dynamic scoping:
[Vars, NumDims] = packageVariables(params, {'n_nrecevier_mod_1412', 'Transpose__5231_0'}, {n_nrecevier_mod_1412, Transpose__5231_0}, [n_nrecevier_mod_1412NumDims Transpose__5231_0NumDims]);
% Call the top-level graph function:
[n_nrecevier_mod_1730, n_nrecevier_mod_1722, n_nrecevier_mod_1721, n_nrecevier_mod_1730NumDims, n_nrecevier_mod_1722NumDims, n_nrecevier_mod_1721NumDims, state] = Shape_To_ReshapeGraph1055(n_nrecevier_mod_1412, Transpose__5231_0, NumDims.n_nrecevier_mod_1412, NumDims.Transpose__5231_0, Vars, NumDims, Training, params.State);
% Postprocess the output data
[n_nrecevier_mod_1730, n_nrecevier_mod_1722, n_nrecevier_mod_1721] = postprocessOutput(n_nrecevier_mod_1730, n_nrecevier_mod_1722, n_nrecevier_mod_1721, outputDataPerms, anyDlarrayInputs, Training, varargin{:});
end

function [n_nrecevier_mod_1730, n_nrecevier_mod_1722, n_nrecevier_mod_1721, n_nrecevier_mod_1730NumDims1062, n_nrecevier_mod_1722NumDims1063, n_nrecevier_mod_1721NumDims1064, state] = Shape_To_ReshapeGraph1055(n_nrecevier_mod_1412, Transpose__5231_0, n_nrecevier_mod_1412NumDims1060, Transpose__5231_0NumDims1061, Vars, NumDims, Training, state)
% Function implementing the graph 'Shape_To_ReshapeGraph1055'
% Update Vars and NumDims from the graph's formal input parameters. Note that state variables are already in Vars.
Vars.n_nrecevier_mod_1412 = n_nrecevier_mod_1412;
NumDims.n_nrecevier_mod_1412 = n_nrecevier_mod_1412NumDims1060;
Vars.Transpose__5231_0 = Transpose__5231_0;
NumDims.Transpose__5231_0 = Transpose__5231_0NumDims1061;

% Execute the operators:
% Shape:
[Vars.Shape__5615_0, NumDims.Shape__5615_0] = onnxShape(Vars.n_nrecevier_mod_1412, NumDims.n_nrecevier_mod_1412, 0, NumDims.n_nrecevier_mod_1412+1);

% Gather:
[Vars.n_nrecevier_mod_1648, NumDims.n_nrecevier_mod_1648] = onnxGather(Vars.Shape__5615_0, Vars.Const__5379, 0, NumDims.Shape__5615_0, NumDims.Const__5379);

% Cast:
if islogical(Vars.n_nrecevier_mod_1648)
    Vars.n_nrecevier_mod_1648 = single(Vars.n_nrecevier_mod_1648);
end
Vars.n_nrecevier_mod_1649 = cast(int32(extractdata(Vars.n_nrecevier_mod_1648)), 'like', Vars.n_nrecevier_mod_1648);
NumDims.n_nrecevier_mod_1649 = NumDims.n_nrecevier_mod_1648;

% Slice:
[Indices, NumDims.n_nrecevier_mod_1668] = prepareSliceArgs(Vars.n_nrecevier_mod_1649, Vars.const__2055, Vars.const_ends__3053, Vars.const_axes__4255, '', NumDims.n_nrecevier_mod_1649);
Vars.n_nrecevier_mod_1668 = subsref(Vars.n_nrecevier_mod_1649, Indices);

% Squeeze:
[Vars.n_nrecevier_mod_1669, NumDims.n_nrecevier_mod_1669] = onnxSqueeze(Vars.n_nrecevier_mod_1668, Vars.const_axes__4255, NumDims.n_nrecevier_mod_1668);

% Div:
Vars.n_nrecevier_mod_1658 = fix(Vars.n_nrecevier_mod_1669 ./ Vars.n_nrecevier_mod_535);
NumDims.n_nrecevier_mod_1658 = max(NumDims.n_nrecevier_mod_1669, NumDims.n_nrecevier_mod_535);

% Unsqueeze:
[shape, NumDims.n_nrecevier_mod_1664] = prepareUnsqueezeArgs(Vars.n_nrecevier_mod_1658, Vars.const_axes__4255, NumDims.n_nrecevier_mod_1658);
Vars.n_nrecevier_mod_1664 = reshape(Vars.n_nrecevier_mod_1658, shape);

% Concat:
[Vars.n_nrecevier_mod_1640, NumDims.n_nrecevier_mod_1640] = onnxConcat(0, {Vars.const_fold_opt__5935, Vars.const_fold_opt__5935, Vars.const_fold_opt__5935, Vars.const_fold_opt__5904, Vars.n_nrecevier_mod_1664}, [NumDims.const_fold_opt__5935, NumDims.const_fold_opt__5935, NumDims.const_fold_opt__5935, NumDims.const_fold_opt__5904, NumDims.n_nrecevier_mod_1664]);

% Cast:
if islogical(Vars.n_nrecevier_mod_1640)
    Vars.n_nrecevier_mod_1640 = single(Vars.n_nrecevier_mod_1640);
end
Vars.n_nrecevier_mod_1642 = cast(int64(extractdata(Vars.n_nrecevier_mod_1640)), 'like', Vars.n_nrecevier_mod_1640);
NumDims.n_nrecevier_mod_1642 = NumDims.n_nrecevier_mod_1640;

% Reshape:
[shape, NumDims.n_nrecevier_mod_1644] = prepareReshapeArgs(Vars.n_nrecevier_mod_1643, Vars.n_nrecevier_mod_1642, NumDims.n_nrecevier_mod_1643, 0);
Vars.n_nrecevier_mod_1644 = reshape(Vars.n_nrecevier_mod_1643, shape{:});

% Reshape:
[shape, NumDims.n_nrecevier_mod_1641] = prepareReshapeArgs(Vars.n_nrecevier_mod_1639, Vars.n_nrecevier_mod_1642, NumDims.n_nrecevier_mod_1639, 0);
Vars.n_nrecevier_mod_1641 = reshape(Vars.n_nrecevier_mod_1639, shape{:});

% Slice:
[Indices, NumDims.n_nrecevier_mod_1667] = prepareSliceArgs(Vars.n_nrecevier_mod_1649, Vars.const_starts__1983, Vars.const__738, Vars.const_axes__4255, '', NumDims.n_nrecevier_mod_1649);
Vars.n_nrecevier_mod_1667 = subsref(Vars.n_nrecevier_mod_1649, Indices);

% Slice:
[Indices, NumDims.n_nrecevier_mod_1666] = prepareSliceArgs(Vars.n_nrecevier_mod_1649, Vars.const_starts__1988, Vars.const_starts__1983, Vars.const_axes__4255, '', NumDims.n_nrecevier_mod_1649);
Vars.n_nrecevier_mod_1666 = subsref(Vars.n_nrecevier_mod_1649, Indices);

% Slice:
[Indices, NumDims.n_nrecevier_mod_1665] = prepareSliceArgs(Vars.n_nrecevier_mod_1649, Vars.const_axes__4255, Vars.const_starts__1988, Vars.const_axes__4255, '', NumDims.n_nrecevier_mod_1649);
Vars.n_nrecevier_mod_1665 = subsref(Vars.n_nrecevier_mod_1649, Indices);

% Concat:
[Vars.n_nrecevier_mod_1663, NumDims.n_nrecevier_mod_1663] = onnxConcat(0, {Vars.n_nrecevier_mod_1665, Vars.n_nrecevier_mod_1666, Vars.n_nrecevier_mod_1667, Vars.const_fold_opt__5904, Vars.n_nrecevier_mod_1664}, [NumDims.n_nrecevier_mod_1665, NumDims.n_nrecevier_mod_1666, NumDims.n_nrecevier_mod_1667, NumDims.const_fold_opt__5904, NumDims.n_nrecevier_mod_1664]);

% Cast:
if islogical(Vars.n_nrecevier_mod_1663)
    Vars.n_nrecevier_mod_1663 = single(Vars.n_nrecevier_mod_1663);
end
Vars.n_nrecevier_mod_1647 = cast(int64(extractdata(Vars.n_nrecevier_mod_1663)), 'like', Vars.n_nrecevier_mod_1663);
NumDims.n_nrecevier_mod_1647 = NumDims.n_nrecevier_mod_1663;

% Reshape:
[shape, NumDims.n_nrecevier_mod_1638] = prepareReshapeArgs(Vars.Transpose__5231_0, Vars.n_nrecevier_mod_1647, NumDims.Transpose__5231_0, 0);
Vars.n_nrecevier_mod_1638 = reshape(Vars.Transpose__5231_0, shape{:});

% ReduceMean:
dims = prepareReduceArgs(Vars.const_fold_opt__5883, NumDims.n_nrecevier_mod_1638);
Vars.n_nrecevier_mod_1661 = mean(Vars.n_nrecevier_mod_1638, dims);
NumDims.n_nrecevier_mod_1661 = NumDims.n_nrecevier_mod_1638;

% Sub:
Vars.n_nrecevier_mod_1659 = Vars.n_nrecevier_mod_1638 - Vars.n_nrecevier_mod_1661;
NumDims.n_nrecevier_mod_1659 = max(NumDims.n_nrecevier_mod_1638, NumDims.n_nrecevier_mod_1661);

% Mul:
Vars.n_nrecevier_mod_1660 = Vars.n_nrecevier_mod_1659 .* Vars.n_nrecevier_mod_1659;
NumDims.n_nrecevier_mod_1660 = max(NumDims.n_nrecevier_mod_1659, NumDims.n_nrecevier_mod_1659);

% ReduceMean:
dims = prepareReduceArgs(Vars.const_fold_opt__5883, NumDims.n_nrecevier_mod_1660);
Vars.n_nrecevier_mod_1662 = mean(Vars.n_nrecevier_mod_1660, dims);
NumDims.n_nrecevier_mod_1662 = NumDims.n_nrecevier_mod_1660;

% Add:
Vars.n_nrecevier_mod_1652 = Vars.n_nrecevier_mod_1662 + Vars.n_nrecevier_mod_146;
NumDims.n_nrecevier_mod_1652 = max(NumDims.n_nrecevier_mod_1662, NumDims.n_nrecevier_mod_146);

% Sqrt:
Vars.n_nrecevier_mod_1650 = sqrt(Vars.n_nrecevier_mod_1652);
NumDims.n_nrecevier_mod_1650 = NumDims.n_nrecevier_mod_1652;

% Reciprocal:
Vars.n_nrecevier_mod_1651 = 1./(Vars.n_nrecevier_mod_1650);
NumDims.n_nrecevier_mod_1651 = NumDims.n_nrecevier_mod_1650;

% Mul:
Vars.n_nrecevier_mod_1654 = Vars.n_nrecevier_mod_1651 .* Vars.n_nrecevier_mod_1641;
NumDims.n_nrecevier_mod_1654 = max(NumDims.n_nrecevier_mod_1651, NumDims.n_nrecevier_mod_1641);

% Mul:
Vars.n_nrecevier_mod_1656 = Vars.n_nrecevier_mod_1661 .* Vars.n_nrecevier_mod_1654;
NumDims.n_nrecevier_mod_1656 = max(NumDims.n_nrecevier_mod_1661, NumDims.n_nrecevier_mod_1654);

% Sub:
Vars.n_nrecevier_mod_1657 = Vars.n_nrecevier_mod_1644 - Vars.n_nrecevier_mod_1656;
NumDims.n_nrecevier_mod_1657 = max(NumDims.n_nrecevier_mod_1644, NumDims.n_nrecevier_mod_1656);

% Mul:
Vars.n_nrecevier_mod_1655 = Vars.n_nrecevier_mod_1638 .* Vars.n_nrecevier_mod_1654;
NumDims.n_nrecevier_mod_1655 = max(NumDims.n_nrecevier_mod_1638, NumDims.n_nrecevier_mod_1654);

% Add:
Vars.n_nrecevier_mod_1653 = Vars.n_nrecevier_mod_1655 + Vars.n_nrecevier_mod_1657;
NumDims.n_nrecevier_mod_1653 = max(NumDims.n_nrecevier_mod_1655, NumDims.n_nrecevier_mod_1657);

% Cast:
if islogical(Vars.n_nrecevier_mod_1649)
    Vars.n_nrecevier_mod_1649 = single(Vars.n_nrecevier_mod_1649);
end
Vars.n_nrecevier_mod_1646 = cast(int64(extractdata(Vars.n_nrecevier_mod_1649)), 'like', Vars.n_nrecevier_mod_1649);
NumDims.n_nrecevier_mod_1646 = NumDims.n_nrecevier_mod_1649;

% Reshape:
[shape, NumDims.n_nrecevier_mod_1645] = prepareReshapeArgs(Vars.n_nrecevier_mod_1653, Vars.n_nrecevier_mod_1646, NumDims.n_nrecevier_mod_1653, 0);
Vars.n_nrecevier_mod_1645 = reshape(Vars.n_nrecevier_mod_1653, shape{:});

% Relu:
Vars.n_nrecevier_mod_1635 = relu(Vars.n_nrecevier_mod_1645);
NumDims.n_nrecevier_mod_1635 = NumDims.n_nrecevier_mod_1645;

% Shape:
[Vars.n_nrecevier_mod_1731, NumDims.n_nrecevier_mod_1731] = onnxShape(Vars.n_nrecevier_mod_1635, NumDims.n_nrecevier_mod_1635, 0, NumDims.n_nrecevier_mod_1635+1);

% PLACEHOLDER FUNCTION FOR UNSUPPORTED OPERATOR (Size):
[Vars.n_nrecevier_mod_1747, NumDims.n_nrecevier_mod_1747] = PLACEHOLDER(Vars.n_nrecevier_mod_1731);

% Shape:
[Vars.n_nrecevier_mod_1728, NumDims.n_nrecevier_mod_1728] = onnxShape(Vars.n_nrecevier_mod_1635, NumDims.n_nrecevier_mod_1635, 0, NumDims.n_nrecevier_mod_1635+1);

% Cast:
if islogical(Vars.n_nrecevier_mod_1728)
    Vars.n_nrecevier_mod_1728 = single(Vars.n_nrecevier_mod_1728);
end
Vars.n_nrecevier_mod_1729 = cast(int32(extractdata(Vars.n_nrecevier_mod_1728)), 'like', Vars.n_nrecevier_mod_1728);
NumDims.n_nrecevier_mod_1729 = NumDims.n_nrecevier_mod_1728;

% Slice:
[Indices, NumDims.n_nrecevier_mod_1779] = prepareSliceArgs(Vars.n_nrecevier_mod_1729, Vars.const_starts__1983, Vars.const__738, Vars.const_axes__4255, '', NumDims.n_nrecevier_mod_1729);
Vars.n_nrecevier_mod_1779 = subsref(Vars.n_nrecevier_mod_1729, Indices);

% Slice:
[Indices, NumDims.n_nrecevier_mod_1778] = prepareSliceArgs(Vars.n_nrecevier_mod_1729, Vars.const_starts__1988, Vars.const_starts__1983, Vars.const_axes__4255, '', NumDims.n_nrecevier_mod_1729);
Vars.n_nrecevier_mod_1778 = subsref(Vars.n_nrecevier_mod_1729, Indices);

% Concat:
[Vars.n_nrecevier_mod_1777, NumDims.n_nrecevier_mod_1777] = onnxConcat(0, {Vars.n_nrecevier_mod_1778, Vars.n_nrecevier_mod_1779}, [NumDims.n_nrecevier_mod_1778, NumDims.n_nrecevier_mod_1779]);

% Add:
Vars.n_nrecevier_mod_1758 = Vars.n_nrecevier_mod_1777 + Vars.n_nrecevier_mod_938;
NumDims.n_nrecevier_mod_1758 = max(NumDims.n_nrecevier_mod_1777, NumDims.n_nrecevier_mod_938);

% Div:
Vars.Div__3946_0 = fix(Vars.n_nrecevier_mod_1758 ./ Vars.n_nrecevier_mod_937);
NumDims.Div__3946_0 = max(NumDims.n_nrecevier_mod_1758, NumDims.n_nrecevier_mod_937);

% Mul:
Vars.Mul__3947_0 = Vars.Div__3946_0 .* Vars.n_nrecevier_mod_937;
NumDims.Mul__3947_0 = max(NumDims.Div__3946_0, NumDims.n_nrecevier_mod_937);

% Sub:
Vars.n_nrecevier_mod_1765 = Vars.n_nrecevier_mod_1758 - Vars.Mul__3947_0;
NumDims.n_nrecevier_mod_1765 = max(NumDims.n_nrecevier_mod_1758, NumDims.Mul__3947_0);

% Sub:
Vars.n_nrecevier_mod_1776 = Vars.n_nrecevier_mod_937 - Vars.n_nrecevier_mod_1765;
NumDims.n_nrecevier_mod_1776 = max(NumDims.n_nrecevier_mod_937, NumDims.n_nrecevier_mod_1765);

% Div:
Vars.Div__3948_0 = fix(Vars.n_nrecevier_mod_1776 ./ Vars.n_nrecevier_mod_937);
NumDims.Div__3948_0 = max(NumDims.n_nrecevier_mod_1776, NumDims.n_nrecevier_mod_937);

% Mul:
Vars.Mul__3949_0 = Vars.Div__3948_0 .* Vars.n_nrecevier_mod_937;
NumDims.Mul__3949_0 = max(NumDims.Div__3948_0, NumDims.n_nrecevier_mod_937);

% Sub:
Vars.n_nrecevier_mod_1766 = Vars.n_nrecevier_mod_1776 - Vars.Mul__3949_0;
NumDims.n_nrecevier_mod_1766 = max(NumDims.n_nrecevier_mod_1776, NumDims.Mul__3949_0);

% Slice:
[Indices, NumDims.n_nrecevier_mod_1775] = prepareSliceArgs(Vars.n_nrecevier_mod_1766, Vars.const_starts__1988, Vars.const_starts__1983, Vars.const_axes__4255, '', NumDims.n_nrecevier_mod_1766);
Vars.n_nrecevier_mod_1775 = subsref(Vars.n_nrecevier_mod_1766, Indices);

% Concat:
[Vars.n_nrecevier_mod_1761, NumDims.n_nrecevier_mod_1761] = onnxConcat(0, {Vars.const_fold_opt__5837, Vars.n_nrecevier_mod_1775}, [NumDims.const_fold_opt__5837, NumDims.n_nrecevier_mod_1775]);

% Unsqueeze:
[shape, NumDims.n_nrecevier_mod_1764] = prepareUnsqueezeArgs(Vars.n_nrecevier_mod_1761, Vars.const_axes__4255, NumDims.n_nrecevier_mod_1761);
Vars.n_nrecevier_mod_1764 = reshape(Vars.n_nrecevier_mod_1761, shape);

% Slice:
[Indices, NumDims.n_nrecevier_mod_1774] = prepareSliceArgs(Vars.n_nrecevier_mod_1766, Vars.const_axes__4255, Vars.const_starts__1988, Vars.const_axes__4255, '', NumDims.n_nrecevier_mod_1766);
Vars.n_nrecevier_mod_1774 = subsref(Vars.n_nrecevier_mod_1766, Indices);

% Concat:
[Vars.n_nrecevier_mod_1760, NumDims.n_nrecevier_mod_1760] = onnxConcat(0, {Vars.const_fold_opt__5837, Vars.n_nrecevier_mod_1774}, [NumDims.const_fold_opt__5837, NumDims.n_nrecevier_mod_1774]);

% Unsqueeze:
[shape, NumDims.n_nrecevier_mod_1763] = prepareUnsqueezeArgs(Vars.n_nrecevier_mod_1760, Vars.const_axes__4255, NumDims.n_nrecevier_mod_1760);
Vars.n_nrecevier_mod_1763 = reshape(Vars.n_nrecevier_mod_1760, shape);

% Concat:
[Vars.n_nrecevier_mod_1762, NumDims.n_nrecevier_mod_1762] = onnxConcat(0, {Vars.n_nrecevier_mod_1763, Vars.n_nrecevier_mod_1764}, [NumDims.n_nrecevier_mod_1763, NumDims.n_nrecevier_mod_1764]);

% Cast:
if islogical(Vars.n_nrecevier_mod_1762)
    Vars.n_nrecevier_mod_1762 = single(Vars.n_nrecevier_mod_1762);
end
Vars.n_nrecevier_mod_1707 = cast(int64(extractdata(Vars.n_nrecevier_mod_1762)), 'like', Vars.n_nrecevier_mod_1762);
NumDims.n_nrecevier_mod_1707 = NumDims.n_nrecevier_mod_1762;

% Transpose:
[perm, NumDims.n_nrecevier_mod_1726] = prepareTransposeArgs('', NumDims.n_nrecevier_mod_1707);
if ~isempty(perm)
    Vars.n_nrecevier_mod_1726 = permute(Vars.n_nrecevier_mod_1707, perm);
end

% Slice:
[Indices, NumDims.n_nrecevier_mod_1720] = prepareSliceArgs(Vars.n_nrecevier_mod_1726, Vars.const__1051, Vars.const__1888, '', '', NumDims.n_nrecevier_mod_1726);
Vars.n_nrecevier_mod_1720 = subsref(Vars.n_nrecevier_mod_1726, Indices);

% Squeeze:
[Vars.n_nrecevier_mod_1722, NumDims.n_nrecevier_mod_1722] = onnxSqueeze(Vars.n_nrecevier_mod_1720, Vars.const_axes__4255, NumDims.n_nrecevier_mod_1720);

% Slice:
[Indices, NumDims.n_nrecevier_mod_1719] = prepareSliceArgs(Vars.n_nrecevier_mod_1726, Vars.const__773, Vars.const__774, '', '', NumDims.n_nrecevier_mod_1726);
Vars.n_nrecevier_mod_1719 = subsref(Vars.n_nrecevier_mod_1726, Indices);

% Squeeze:
[Vars.n_nrecevier_mod_1721, NumDims.n_nrecevier_mod_1721] = onnxSqueeze(Vars.n_nrecevier_mod_1719, Vars.const_axes__4255, NumDims.n_nrecevier_mod_1719);

% Add:
Vars.n_nrecevier_mod_1759 = Vars.n_nrecevier_mod_937 + Vars.n_nrecevier_mod_1766;
NumDims.n_nrecevier_mod_1759 = max(NumDims.n_nrecevier_mod_937, NumDims.n_nrecevier_mod_1766);

% Slice:
[Indices, NumDims.n_nrecevier_mod_1773] = prepareSliceArgs(Vars.n_nrecevier_mod_1759, Vars.const_starts__1988, Vars.const_starts__1983, Vars.const_axes__4255, '', NumDims.n_nrecevier_mod_1759);
Vars.n_nrecevier_mod_1773 = subsref(Vars.n_nrecevier_mod_1759, Indices);

% Concat:
[Vars.n_nrecevier_mod_1768, NumDims.n_nrecevier_mod_1768] = onnxConcat(0, {Vars.const_fold_opt__5913, Vars.n_nrecevier_mod_1773}, [NumDims.const_fold_opt__5913, NumDims.n_nrecevier_mod_1773]);

% Unsqueeze:
[shape, NumDims.n_nrecevier_mod_1771] = prepareUnsqueezeArgs(Vars.n_nrecevier_mod_1768, Vars.const_axes__4255, NumDims.n_nrecevier_mod_1768);
Vars.n_nrecevier_mod_1771 = reshape(Vars.n_nrecevier_mod_1768, shape);

% Slice:
[Indices, NumDims.n_nrecevier_mod_1772] = prepareSliceArgs(Vars.n_nrecevier_mod_1759, Vars.const_axes__4255, Vars.const_starts__1988, Vars.const_axes__4255, '', NumDims.n_nrecevier_mod_1759);
Vars.n_nrecevier_mod_1772 = subsref(Vars.n_nrecevier_mod_1759, Indices);

% Concat:
[Vars.n_nrecevier_mod_1767, NumDims.n_nrecevier_mod_1767] = onnxConcat(0, {Vars.const_fold_opt__5912, Vars.n_nrecevier_mod_1772}, [NumDims.const_fold_opt__5912, NumDims.n_nrecevier_mod_1772]);

% Unsqueeze:
[shape, NumDims.n_nrecevier_mod_1770] = prepareUnsqueezeArgs(Vars.n_nrecevier_mod_1767, Vars.const_axes__4255, NumDims.n_nrecevier_mod_1767);
Vars.n_nrecevier_mod_1770 = reshape(Vars.n_nrecevier_mod_1767, shape);

% Concat:
[Vars.n_nrecevier_mod_1769, NumDims.n_nrecevier_mod_1769] = onnxConcat(0, {Vars.n_nrecevier_mod_1770, Vars.n_nrecevier_mod_1771}, [NumDims.n_nrecevier_mod_1770, NumDims.n_nrecevier_mod_1771]);

% Cast:
if islogical(Vars.n_nrecevier_mod_1769)
    Vars.n_nrecevier_mod_1769 = single(Vars.n_nrecevier_mod_1769);
end
Vars.n_nrecevier_mod_1732 = cast(int64(extractdata(Vars.n_nrecevier_mod_1769)), 'like', Vars.n_nrecevier_mod_1769);
NumDims.n_nrecevier_mod_1732 = NumDims.n_nrecevier_mod_1769;

% Shape:
[Vars.n_nrecevier_mod_1745, NumDims.n_nrecevier_mod_1745] = onnxShape(Vars.n_nrecevier_mod_1732, NumDims.n_nrecevier_mod_1732, 0, NumDims.n_nrecevier_mod_1732+1);

% Slice:
[Indices, NumDims.n_nrecevier_mod_1748] = prepareSliceArgs(Vars.n_nrecevier_mod_1745, Vars.const_axes__4255, Vars.const_starts__1988, '', '', NumDims.n_nrecevier_mod_1745);
Vars.n_nrecevier_mod_1748 = subsref(Vars.n_nrecevier_mod_1745, Indices);

% Sub:
Vars.n_nrecevier_mod_1751 = Vars.n_nrecevier_mod_1747 - Vars.n_nrecevier_mod_1748;
NumDims.n_nrecevier_mod_1751 = max(NumDims.n_nrecevier_mod_1747, NumDims.n_nrecevier_mod_1748);

% Sub:
Vars.n_nrecevier_mod_1752 = Vars.n_nrecevier_mod_1751 - Vars.const_starts__1988;
NumDims.n_nrecevier_mod_1752 = max(NumDims.n_nrecevier_mod_1751, NumDims.const_starts__1988);

% Mul:
Vars.n_nrecevier_mod_1737 = Vars.const__1231 .* Vars.n_nrecevier_mod_1752;
NumDims.n_nrecevier_mod_1737 = max(NumDims.const__1231, NumDims.n_nrecevier_mod_1752);

% Pad:
[Vars.n_nrecevier_mod_1738, NumDims.n_nrecevier_mod_1738] = onnxPad(Vars.n_nrecevier_mod_1732, Vars.const__1230, 0, 'constant', dlarray([0:NumDims.n_nrecevier_mod_1732]'), NumDims.n_nrecevier_mod_1732);

% Pad:
[Vars.n_nrecevier_mod_1739, NumDims.n_nrecevier_mod_1739] = onnxPad(Vars.n_nrecevier_mod_1738, Vars.n_nrecevier_mod_1737, 0, 'constant', dlarray([0:NumDims.n_nrecevier_mod_1738]'), NumDims.n_nrecevier_mod_1738);

% Transpose:
[perm, NumDims.n_nrecevier_mod_1753] = prepareTransposeArgs('', NumDims.n_nrecevier_mod_1739);
if ~isempty(perm)
    Vars.n_nrecevier_mod_1753 = permute(Vars.n_nrecevier_mod_1739, perm);
end

% Reshape:
[shape, NumDims.n_nrecevier_mod_1741] = prepareReshapeArgs(Vars.n_nrecevier_mod_1753, Vars.const__2055, NumDims.n_nrecevier_mod_1753, 0);
Vars.n_nrecevier_mod_1741 = reshape(Vars.n_nrecevier_mod_1753, shape{:});

% Pad:
[Vars.n_nrecevier_mod_1740, NumDims.n_nrecevier_mod_1740] = onnxPad(Vars.n_nrecevier_mod_1635, Vars.n_nrecevier_mod_1741, 0, 'constant', dlarray([0:NumDims.n_nrecevier_mod_1635]'), NumDims.n_nrecevier_mod_1635);

% Shape:
[Vars.n_nrecevier_mod_1746, NumDims.n_nrecevier_mod_1746] = onnxShape(Vars.n_nrecevier_mod_1740, NumDims.n_nrecevier_mod_1740, 0, NumDims.n_nrecevier_mod_1740+1);

% Slice:
[Indices, NumDims.n_nrecevier_mod_1750] = prepareSliceArgs(Vars.n_nrecevier_mod_1746, Vars.const__738, Vars.const__664, '', '', NumDims.n_nrecevier_mod_1746);
Vars.n_nrecevier_mod_1750 = subsref(Vars.n_nrecevier_mod_1746, Indices);

% Slice:
[Indices, NumDims.n_nrecevier_mod_1749] = prepareSliceArgs(Vars.n_nrecevier_mod_1746, Vars.const_starts__1988, Vars.const__738, '', '', NumDims.n_nrecevier_mod_1746);
Vars.n_nrecevier_mod_1749 = subsref(Vars.n_nrecevier_mod_1746, Indices);

% Div:
Vars.n_nrecevier_mod_1736 = fix(Vars.n_nrecevier_mod_1749 ./ Vars.const__739);
NumDims.n_nrecevier_mod_1736 = max(NumDims.n_nrecevier_mod_1749, NumDims.const__739);

% Concat:
[Vars.n_nrecevier_mod_1735, NumDims.n_nrecevier_mod_1735] = onnxConcat(0, {Vars.const__2055, Vars.n_nrecevier_mod_1736, Vars.n_nrecevier_mod_1750}, [NumDims.const__2055, NumDims.n_nrecevier_mod_1736, NumDims.n_nrecevier_mod_1750]);

% Concat:
[Vars.n_nrecevier_mod_1733, NumDims.n_nrecevier_mod_1733] = onnxConcat(0, {Vars.n_nrecevier_mod_1736, Vars.const__739}, [NumDims.n_nrecevier_mod_1736, NumDims.const__739]);

% Reshape:
[shape, NumDims.n_nrecevier_mod_1742] = prepareReshapeArgs(Vars.n_nrecevier_mod_1733, Vars.const__1228, NumDims.n_nrecevier_mod_1733, 0);
Vars.n_nrecevier_mod_1742 = reshape(Vars.n_nrecevier_mod_1733, shape{:});

% Transpose:
[perm, NumDims.n_nrecevier_mod_1754] = prepareTransposeArgs('', NumDims.n_nrecevier_mod_1742);
if ~isempty(perm)
    Vars.n_nrecevier_mod_1754 = permute(Vars.n_nrecevier_mod_1742, perm);
end

% Reshape:
[shape, NumDims.n_nrecevier_mod_1743] = prepareReshapeArgs(Vars.n_nrecevier_mod_1754, Vars.const__2055, NumDims.n_nrecevier_mod_1754, 0);
Vars.n_nrecevier_mod_1743 = reshape(Vars.n_nrecevier_mod_1754, shape{:});

% Concat:
[Vars.n_nrecevier_mod_1734, NumDims.n_nrecevier_mod_1734] = onnxConcat(0, {Vars.const__2055, Vars.n_nrecevier_mod_1743, Vars.n_nrecevier_mod_1750}, [NumDims.const__2055, NumDims.n_nrecevier_mod_1743, NumDims.n_nrecevier_mod_1750]);

% Reshape:
[shape, NumDims.n_nrecevier_mod_1744] = prepareReshapeArgs(Vars.n_nrecevier_mod_1740, Vars.n_nrecevier_mod_1734, NumDims.n_nrecevier_mod_1740, 0);
Vars.n_nrecevier_mod_1744 = reshape(Vars.n_nrecevier_mod_1740, shape{:});

% Transpose:
[perm, NumDims.n_nrecevier_mod_1755] = prepareTransposeArgs(Vars.TransposePerm1059, NumDims.n_nrecevier_mod_1744);
if ~isempty(perm)
    Vars.n_nrecevier_mod_1755 = permute(Vars.n_nrecevier_mod_1744, perm);
end

% Reshape:
[shape, NumDims.n_nrecevier_mod_1730] = prepareReshapeArgs(Vars.n_nrecevier_mod_1755, Vars.n_nrecevier_mod_1735, NumDims.n_nrecevier_mod_1755, 0);
Vars.n_nrecevier_mod_1730 = reshape(Vars.n_nrecevier_mod_1755, shape{:});

% Set graph output arguments from Vars and NumDims:
n_nrecevier_mod_1730 = Vars.n_nrecevier_mod_1730;
n_nrecevier_mod_1730NumDims1062 = NumDims.n_nrecevier_mod_1730;
n_nrecevier_mod_1722 = Vars.n_nrecevier_mod_1722;
n_nrecevier_mod_1722NumDims1063 = NumDims.n_nrecevier_mod_1722;
n_nrecevier_mod_1721 = Vars.n_nrecevier_mod_1721;
n_nrecevier_mod_1721NumDims1064 = NumDims.n_nrecevier_mod_1721;
% Set output state from Vars:
state = updateStruct(state, Vars);
end

function [inputDataPerms, outputDataPerms, Training] = parseInputs(n_nrecevier_mod_1412, Transpose__5231_0, numDataOutputs, params, varargin)
% Function to validate inputs to Shape_To_ReshapeFcn:
p = inputParser;
isValidArrayInput = @(x)isnumeric(x) || isstring(x);
isValidONNXParameters = @(x)isa(x, 'ONNXParameters');
addRequired(p, 'n_nrecevier_mod_1412', isValidArrayInput);
addRequired(p, 'Transpose__5231_0', isValidArrayInput);
addRequired(p, 'params', isValidONNXParameters);
addParameter(p, 'InputDataPermutation', 'auto');
addParameter(p, 'OutputDataPermutation', 'auto');
addParameter(p, 'Training', false);
parse(p, n_nrecevier_mod_1412, Transpose__5231_0, params, varargin{:});
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

function [n_nrecevier_mod_1412, Transpose__5231_0, Training, outputDataPerms, anyDlarrayInputs] = preprocessInput(n_nrecevier_mod_1412, Transpose__5231_0, params, varargin)
% Parse input arguments
[inputDataPerms, outputDataPerms, Training] = parseInputs(n_nrecevier_mod_1412, Transpose__5231_0, 3, params, varargin{:});
anyDlarrayInputs = any(cellfun(@(x)isa(x, 'dlarray'), {n_nrecevier_mod_1412, Transpose__5231_0}));
% Make the input variables into unlabelled dlarrays:
n_nrecevier_mod_1412 = makeUnlabeledDlarray(n_nrecevier_mod_1412);
Transpose__5231_0 = makeUnlabeledDlarray(Transpose__5231_0);
% Permute inputs if requested:
n_nrecevier_mod_1412 = permuteInputVar(n_nrecevier_mod_1412, inputDataPerms{1}, 4);
Transpose__5231_0 = permuteInputVar(Transpose__5231_0, inputDataPerms{2}, 4);
end

function [n_nrecevier_mod_1730, n_nrecevier_mod_1722, n_nrecevier_mod_1721] = postprocessOutput(n_nrecevier_mod_1730, n_nrecevier_mod_1722, n_nrecevier_mod_1721, outputDataPerms, anyDlarrayInputs, Training, varargin)
% Set output type:
if ~anyDlarrayInputs && ~Training
    if isdlarray(n_nrecevier_mod_1730)
        n_nrecevier_mod_1730 = extractdata(n_nrecevier_mod_1730);
    end
    if isdlarray(n_nrecevier_mod_1722)
        n_nrecevier_mod_1722 = extractdata(n_nrecevier_mod_1722);
    end
    if isdlarray(n_nrecevier_mod_1721)
        n_nrecevier_mod_1721 = extractdata(n_nrecevier_mod_1721);
    end
end
% Permute outputs if requested:
n_nrecevier_mod_1730 = permuteOutputVar(n_nrecevier_mod_1730, outputDataPerms{1}, 4);
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
