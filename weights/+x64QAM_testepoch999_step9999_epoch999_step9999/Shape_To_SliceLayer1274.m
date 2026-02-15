classdef Shape_To_SliceLayer1274 < nnet.layer.Layer & nnet.layer.Formattable
    % A custom layer auto-generated while importing an ONNX network.

    %#codegen
    %#ok<*PROPLC>
    %#ok<*NBRAK>
    %#ok<*INUSL>
    %#ok<*VARARG>

    properties (Learnable)
    end

    properties
        ONNXParams         % An ONNXParameters object containing parameters used by this layer.
    end

    methods
        function this = Shape_To_SliceLayer1274(name, onnxParams)
            this.Name = name;
            this.NumInputs = 6;
            this.OutputNames = {'n_nrecevier_mod_581'};
            this.ONNXParams = onnxParams;
        end

        function [n_nrecevier_mod_581] = predict(this, n_nrecevier_mod_632, n_nrecevier_mod_598, Transpose__4955_0, n_nrecevier_mod_597, n_nrecevier_mod_598NumDims, n_nrecevier_mod_597NumDims)
            if isdlarray(n_nrecevier_mod_632)
                n_nrecevier_mod_632 = stripdims(n_nrecevier_mod_632);
            end
            if isdlarray(n_nrecevier_mod_598)
                n_nrecevier_mod_598 = stripdims(n_nrecevier_mod_598);
            end
            if isdlarray(Transpose__4955_0)
                Transpose__4955_0 = stripdims(Transpose__4955_0);
            end
            if isdlarray(n_nrecevier_mod_597)
                n_nrecevier_mod_597 = stripdims(n_nrecevier_mod_597);
            end
            n_nrecevier_mod_632NumDims = 4;
            n_nrecevier_mod_598NumDims = numel(n_nrecevier_mod_598NumDims);
            Transpose__4955_0NumDims = 4;
            n_nrecevier_mod_597NumDims = numel(n_nrecevier_mod_597NumDims);
            onnxParams = this.ONNXParams;
            [n_nrecevier_mod_581, n_nrecevier_mod_581NumDims] = Shape_To_SliceFcn(n_nrecevier_mod_632, n_nrecevier_mod_598, Transpose__4955_0, n_nrecevier_mod_597, n_nrecevier_mod_632NumDims, n_nrecevier_mod_598NumDims, Transpose__4955_0NumDims, n_nrecevier_mod_597NumDims, onnxParams, 'Training', false, ...
                'InputDataPermutation', {[4 3 1 2], ['as-is'], [4 1 2 3], ['as-is'], ['as-is'], ['as-is'], ['as-is'], ['as-is']}, ...
                'OutputDataPermutation', {[2 3 4 1], ['as-is']});
            if any(cellfun(@(A)~isnumeric(A) && ~islogical(A), {n_nrecevier_mod_581}))
                fprintf('Runtime error in network. At least one output of custom layer ''%s'' is a non-numeric, non-logical value.\n', 'Shape_To_SliceLayer1274');
                error(message('nnet_cnn_onnx:onnx:BadCustomLayerRuntimeOutput', 'Shape_To_SliceLayer1274'));
            end
            n_nrecevier_mod_581 = dlarray(single(n_nrecevier_mod_581), 'SSCB');
            if ~coder.target('MATLAB')
                n_nrecevier_mod_581 = extractdata(n_nrecevier_mod_581);
            end
        end

        function [n_nrecevier_mod_581] = forward(this, n_nrecevier_mod_632, n_nrecevier_mod_598, Transpose__4955_0, n_nrecevier_mod_597, n_nrecevier_mod_598NumDims, n_nrecevier_mod_597NumDims)
            if isdlarray(n_nrecevier_mod_632)
                n_nrecevier_mod_632 = stripdims(n_nrecevier_mod_632);
            end
            if isdlarray(n_nrecevier_mod_598)
                n_nrecevier_mod_598 = stripdims(n_nrecevier_mod_598);
            end
            if isdlarray(Transpose__4955_0)
                Transpose__4955_0 = stripdims(Transpose__4955_0);
            end
            if isdlarray(n_nrecevier_mod_597)
                n_nrecevier_mod_597 = stripdims(n_nrecevier_mod_597);
            end
            n_nrecevier_mod_632NumDims = 4;
            n_nrecevier_mod_598NumDims = numel(n_nrecevier_mod_598NumDims);
            Transpose__4955_0NumDims = 4;
            n_nrecevier_mod_597NumDims = numel(n_nrecevier_mod_597NumDims);
            onnxParams = this.ONNXParams;
            [n_nrecevier_mod_581, n_nrecevier_mod_581NumDims] = Shape_To_SliceFcn(n_nrecevier_mod_632, n_nrecevier_mod_598, Transpose__4955_0, n_nrecevier_mod_597, n_nrecevier_mod_632NumDims, n_nrecevier_mod_598NumDims, Transpose__4955_0NumDims, n_nrecevier_mod_597NumDims, onnxParams, 'Training', true, ...
                'InputDataPermutation', {[4 3 1 2], ['as-is'], [4 1 2 3], ['as-is'], ['as-is'], ['as-is'], ['as-is'], ['as-is']}, ...
                'OutputDataPermutation', {[2 3 4 1], ['as-is']});
            if any(cellfun(@(A)~isnumeric(A) && ~islogical(A), {n_nrecevier_mod_581}))
                fprintf('Runtime error in network. At least one output of custom layer ''%s'' is a non-numeric, non-logical value.\n', 'Shape_To_SliceLayer1274');
                error(message('nnet_cnn_onnx:onnx:BadCustomLayerRuntimeOutput', 'Shape_To_SliceLayer1274'));
            end
            n_nrecevier_mod_581 = dlarray(single(n_nrecevier_mod_581), 'SSCB');
            if ~coder.target('MATLAB')
                n_nrecevier_mod_581 = extractdata(n_nrecevier_mod_581);
            end
        end
    end
end

function [n_nrecevier_mod_581, n_nrecevier_mod_581NumDims, state] = Shape_To_SliceFcn(n_nrecevier_mod_632, n_nrecevier_mod_598, Transpose__4955_0, n_nrecevier_mod_597, n_nrecevier_mod_632NumDims, n_nrecevier_mod_598NumDims, Transpose__4955_0NumDims, n_nrecevier_mod_597NumDims, params, varargin)
%SHAPE_TO_SLICEFCN Function implementing an imported ONNX network.
%
% THIS FILE WAS AUTO-GENERATED BY importONNXFunction.
% ONNX Operator Set Version: 18
%
% Variable names in this function are taken from the original ONNX file.
%
% [N_NRECEVIER_MOD_581] = Shape_To_SliceFcn(N_NRECEVIER_MOD_632, N_NRECEVIER_MOD_598, TRANSPOSE__4955_0, N_NRECEVIER_MOD_597, PARAMS)
%			- Evaluates the imported ONNX network SHAPE_TO_SLICEFCN with input(s)
%			N_NRECEVIER_MOD_632, N_NRECEVIER_MOD_598, TRANSPOSE__4955_0, N_NRECEVIER_MOD_597 and the imported network parameters in PARAMS. Returns
%			network output(s) in N_NRECEVIER_MOD_581.
%
% [N_NRECEVIER_MOD_581, STATE] = Shape_To_SliceFcn(N_NRECEVIER_MOD_632, N_NRECEVIER_MOD_598, TRANSPOSE__4955_0, N_NRECEVIER_MOD_597, PARAMS)
%			- Additionally returns state variables in STATE. When training,
%			use this form and set TRAINING to true.
%
% [__] = Shape_To_SliceFcn(N_NRECEVIER_MOD_632, N_NRECEVIER_MOD_598, TRANSPOSE__4955_0, N_NRECEVIER_MOD_597, PARAMS, 'NAME1', VAL1, 'NAME2', VAL2, ...)
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
% N_NRECEVIER_MOD_632, N_NRECEVIER_MOD_598, TRANSPOSE__4955_0, N_NRECEVIER_MOD_597
%			- Input(s) to the ONNX network.
%			  The input size(s) expected by the ONNX file are:
%				  N_NRECEVIER_MOD_632:		[Unknown, Unknown, Unknown, Unknown]				Type: FLOAT
%				  N_NRECEVIER_MOD_598:		[1, 1]				Type: FLOAT
%				  TRANSPOSE__4955_0:		[Unknown, Unknown, Unknown, Unknown]				Type: FLOAT
%				  N_NRECEVIER_MOD_597:		[1, 1]				Type: FLOAT
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
% N_NRECEVIER_MOD_581
%			- Output(s) of the ONNX network.
%			  Without permutation, the size(s) of the outputs are:
%				  N_NRECEVIER_MOD_581:		[Unknown, Unknown, Unknown, Unknown]				Type: FLOAT
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
[n_nrecevier_mod_632, n_nrecevier_mod_598, Transpose__4955_0, n_nrecevier_mod_597, Training, outputDataPerms, anyDlarrayInputs] = preprocessInput(n_nrecevier_mod_632, n_nrecevier_mod_598, Transpose__4955_0, n_nrecevier_mod_597, params, varargin{:});
% Put all variables into a single struct to implement dynamic scoping:
[Vars, NumDims] = packageVariables(params, {'n_nrecevier_mod_632', 'n_nrecevier_mod_598', 'Transpose__4955_0', 'n_nrecevier_mod_597'}, {n_nrecevier_mod_632, n_nrecevier_mod_598, Transpose__4955_0, n_nrecevier_mod_597}, [n_nrecevier_mod_632NumDims n_nrecevier_mod_598NumDims Transpose__4955_0NumDims n_nrecevier_mod_597NumDims]);
% Call the top-level graph function:
[n_nrecevier_mod_581, n_nrecevier_mod_581NumDims, state] = Shape_To_SliceGraph1266(n_nrecevier_mod_632, n_nrecevier_mod_598, Transpose__4955_0, n_nrecevier_mod_597, NumDims.n_nrecevier_mod_632, NumDims.n_nrecevier_mod_598, NumDims.Transpose__4955_0, NumDims.n_nrecevier_mod_597, Vars, NumDims, Training, params.State);
% Postprocess the output data
[n_nrecevier_mod_581] = postprocessOutput(n_nrecevier_mod_581, outputDataPerms, anyDlarrayInputs, Training, varargin{:});
end

function [n_nrecevier_mod_581, n_nrecevier_mod_581NumDims1273, state] = Shape_To_SliceGraph1266(n_nrecevier_mod_632, n_nrecevier_mod_598, Transpose__4955_0, n_nrecevier_mod_597, n_nrecevier_mod_632NumDims1269, n_nrecevier_mod_598NumDims1270, Transpose__4955_0NumDims1271, n_nrecevier_mod_597NumDims1272, Vars, NumDims, Training, state)
% Function implementing the graph 'Shape_To_SliceGraph1266'
% Update Vars and NumDims from the graph's formal input parameters. Note that state variables are already in Vars.
Vars.n_nrecevier_mod_632 = n_nrecevier_mod_632;
NumDims.n_nrecevier_mod_632 = n_nrecevier_mod_632NumDims1269;
Vars.n_nrecevier_mod_598 = n_nrecevier_mod_598;
NumDims.n_nrecevier_mod_598 = n_nrecevier_mod_598NumDims1270;
Vars.Transpose__4955_0 = Transpose__4955_0;
NumDims.Transpose__4955_0 = Transpose__4955_0NumDims1271;
Vars.n_nrecevier_mod_597 = n_nrecevier_mod_597;
NumDims.n_nrecevier_mod_597 = n_nrecevier_mod_597NumDims1272;

% Execute the operators:
% Shape:
[Vars.Shape__5447_0, NumDims.Shape__5447_0] = onnxShape(Vars.n_nrecevier_mod_632, NumDims.n_nrecevier_mod_632, 0, NumDims.n_nrecevier_mod_632+1);

% Gather:
[Vars.n_nrecevier_mod_582, NumDims.n_nrecevier_mod_582] = onnxGather(Vars.Shape__5447_0, Vars.Const__5379, 0, NumDims.Shape__5447_0, NumDims.Const__5379);

% Slice:
[Indices, NumDims.n_nrecevier_mod_594] = prepareSliceArgs(Vars.n_nrecevier_mod_582, Vars.const__738, Vars.const__664, '', '', NumDims.n_nrecevier_mod_582);
Vars.n_nrecevier_mod_594 = subsref(Vars.n_nrecevier_mod_582, Indices);

% Slice:
[Indices, NumDims.n_nrecevier_mod_593] = prepareSliceArgs(Vars.n_nrecevier_mod_582, Vars.const_starts__1988, Vars.const__738, '', '', NumDims.n_nrecevier_mod_582);
Vars.n_nrecevier_mod_593 = subsref(Vars.n_nrecevier_mod_582, Indices);

% Mul:
Vars.n_nrecevier_mod_588 = Vars.n_nrecevier_mod_593 .* Vars.const__739;
NumDims.n_nrecevier_mod_588 = max(NumDims.n_nrecevier_mod_593, NumDims.const__739);

% Sub:
Vars.n_nrecevier_mod_599 = Vars.n_nrecevier_mod_588 - Vars.n_nrecevier_mod_598;
NumDims.n_nrecevier_mod_599 = max(NumDims.n_nrecevier_mod_588, NumDims.n_nrecevier_mod_598);

% Concat:
[Vars.n_nrecevier_mod_586, NumDims.n_nrecevier_mod_586] = onnxConcat(0, {Vars.const__2055, Vars.n_nrecevier_mod_588, Vars.n_nrecevier_mod_594}, [NumDims.const__2055, NumDims.n_nrecevier_mod_588, NumDims.n_nrecevier_mod_594]);

% Concat:
[Vars.n_nrecevier_mod_584, NumDims.n_nrecevier_mod_584] = onnxConcat(0, {Vars.n_nrecevier_mod_593, Vars.const__739}, [NumDims.n_nrecevier_mod_593, NumDims.const__739]);

% Reshape:
[shape, NumDims.n_nrecevier_mod_589] = prepareReshapeArgs(Vars.n_nrecevier_mod_584, Vars.const__1228, NumDims.n_nrecevier_mod_584, 0);
Vars.n_nrecevier_mod_589 = reshape(Vars.n_nrecevier_mod_584, shape{:});

% Transpose:
[perm, NumDims.n_nrecevier_mod_600] = prepareTransposeArgs('', NumDims.n_nrecevier_mod_589);
if ~isempty(perm)
    Vars.n_nrecevier_mod_600 = permute(Vars.n_nrecevier_mod_589, perm);
end

% Reshape:
[shape, NumDims.n_nrecevier_mod_590] = prepareReshapeArgs(Vars.n_nrecevier_mod_600, Vars.const__2055, NumDims.n_nrecevier_mod_600, 0);
Vars.n_nrecevier_mod_590 = reshape(Vars.n_nrecevier_mod_600, shape{:});

% Concat:
[Vars.n_nrecevier_mod_585, NumDims.n_nrecevier_mod_585] = onnxConcat(0, {Vars.const__2055, Vars.n_nrecevier_mod_590, Vars.n_nrecevier_mod_594}, [NumDims.const__2055, NumDims.n_nrecevier_mod_590, NumDims.n_nrecevier_mod_594]);

% Gather:
[Vars.n_nrecevier_mod_587, NumDims.n_nrecevier_mod_587] = onnxGather(Vars.n_nrecevier_mod_585, Vars.const__752, 0, NumDims.n_nrecevier_mod_585, NumDims.const__752);

% Reshape:
[shape, NumDims.n_nrecevier_mod_591] = prepareReshapeArgs(Vars.Transpose__4955_0, Vars.n_nrecevier_mod_587, NumDims.Transpose__4955_0, 0);
Vars.n_nrecevier_mod_591 = reshape(Vars.Transpose__4955_0, shape{:});

% Transpose:
[perm, NumDims.n_nrecevier_mod_601] = prepareTransposeArgs(Vars.TransposePerm1268, NumDims.n_nrecevier_mod_591);
if ~isempty(perm)
    Vars.n_nrecevier_mod_601 = permute(Vars.n_nrecevier_mod_591, perm);
end

% Reshape:
[shape, NumDims.n_nrecevier_mod_592] = prepareReshapeArgs(Vars.n_nrecevier_mod_601, Vars.n_nrecevier_mod_586, NumDims.n_nrecevier_mod_601, 0);
Vars.n_nrecevier_mod_592 = reshape(Vars.n_nrecevier_mod_601, shape{:});

% Slice:
[Indices, NumDims.n_nrecevier_mod_581] = prepareSliceArgs(Vars.n_nrecevier_mod_592, Vars.n_nrecevier_mod_597, Vars.n_nrecevier_mod_599, Vars.const__1889, '', NumDims.n_nrecevier_mod_592);
Vars.n_nrecevier_mod_581 = subsref(Vars.n_nrecevier_mod_592, Indices);

% Set graph output arguments from Vars and NumDims:
n_nrecevier_mod_581 = Vars.n_nrecevier_mod_581;
n_nrecevier_mod_581NumDims1273 = NumDims.n_nrecevier_mod_581;
% Set output state from Vars:
state = updateStruct(state, Vars);
end

function [inputDataPerms, outputDataPerms, Training] = parseInputs(n_nrecevier_mod_632, n_nrecevier_mod_598, Transpose__4955_0, n_nrecevier_mod_597, numDataOutputs, params, varargin)
% Function to validate inputs to Shape_To_SliceFcn:
p = inputParser;
isValidArrayInput = @(x)isnumeric(x) || isstring(x);
isValidONNXParameters = @(x)isa(x, 'ONNXParameters');
addRequired(p, 'n_nrecevier_mod_632', isValidArrayInput);
addRequired(p, 'n_nrecevier_mod_598', isValidArrayInput);
addRequired(p, 'Transpose__4955_0', isValidArrayInput);
addRequired(p, 'n_nrecevier_mod_597', isValidArrayInput);
addRequired(p, 'params', isValidONNXParameters);
addParameter(p, 'InputDataPermutation', 'auto');
addParameter(p, 'OutputDataPermutation', 'auto');
addParameter(p, 'Training', false);
parse(p, n_nrecevier_mod_632, n_nrecevier_mod_598, Transpose__4955_0, n_nrecevier_mod_597, params, varargin{:});
inputDataPerms = p.Results.InputDataPermutation;
outputDataPerms = p.Results.OutputDataPermutation;
Training = p.Results.Training;
if isnumeric(inputDataPerms)
    inputDataPerms = {inputDataPerms};
end
if isstring(inputDataPerms) && isscalar(inputDataPerms) || ischar(inputDataPerms)
    inputDataPerms = repmat({inputDataPerms},1,4);
end
if isnumeric(outputDataPerms)
    outputDataPerms = {outputDataPerms};
end
if isstring(outputDataPerms) && isscalar(outputDataPerms) || ischar(outputDataPerms)
    outputDataPerms = repmat({outputDataPerms},1,numDataOutputs);
end
end

function [n_nrecevier_mod_632, n_nrecevier_mod_598, Transpose__4955_0, n_nrecevier_mod_597, Training, outputDataPerms, anyDlarrayInputs] = preprocessInput(n_nrecevier_mod_632, n_nrecevier_mod_598, Transpose__4955_0, n_nrecevier_mod_597, params, varargin)
% Parse input arguments
[inputDataPerms, outputDataPerms, Training] = parseInputs(n_nrecevier_mod_632, n_nrecevier_mod_598, Transpose__4955_0, n_nrecevier_mod_597, 1, params, varargin{:});
anyDlarrayInputs = any(cellfun(@(x)isa(x, 'dlarray'), {n_nrecevier_mod_632, n_nrecevier_mod_598, Transpose__4955_0, n_nrecevier_mod_597}));
% Make the input variables into unlabelled dlarrays:
n_nrecevier_mod_632 = makeUnlabeledDlarray(n_nrecevier_mod_632);
n_nrecevier_mod_598 = makeUnlabeledDlarray(n_nrecevier_mod_598);
Transpose__4955_0 = makeUnlabeledDlarray(Transpose__4955_0);
n_nrecevier_mod_597 = makeUnlabeledDlarray(n_nrecevier_mod_597);
% Permute inputs if requested:
n_nrecevier_mod_632 = permuteInputVar(n_nrecevier_mod_632, inputDataPerms{1}, 4);
n_nrecevier_mod_598 = permuteInputVar(n_nrecevier_mod_598, inputDataPerms{2}, 0);
Transpose__4955_0 = permuteInputVar(Transpose__4955_0, inputDataPerms{3}, 4);
n_nrecevier_mod_597 = permuteInputVar(n_nrecevier_mod_597, inputDataPerms{4}, 0);
end

function [n_nrecevier_mod_581] = postprocessOutput(n_nrecevier_mod_581, outputDataPerms, anyDlarrayInputs, Training, varargin)
% Set output type:
if ~anyDlarrayInputs && ~Training
    if isdlarray(n_nrecevier_mod_581)
        n_nrecevier_mod_581 = extractdata(n_nrecevier_mod_581);
    end
end
% Permute outputs if requested:
n_nrecevier_mod_581 = permuteOutputVar(n_nrecevier_mod_581, outputDataPerms{1}, 4);
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
