classdef Shape_To_ReshapeLayer1145 < nnet.layer.Layer & nnet.layer.Formattable
    % A custom layer auto-generated while importing an ONNX network.

    %#codegen
    %#ok<*PROPLC>
    %#ok<*NBRAK>
    %#ok<*INUSL>
    %#ok<*VARARG>

    properties (Learnable)
        const_fold_opt__5883
        n_nrecevier_mod_1190
        n_nrecevier_mod_1192
    end

    properties
        ONNXParams         % An ONNXParameters object containing parameters used by this layer.
    end

    methods
        function this = Shape_To_ReshapeLayer1145(name, onnxParams)
            this.Name = name;
            this.NumInputs = 2;
            this.NumOutputs = 5;
            this.OutputNames = {'n_nrecevier_mod_1282', 'n_nrecevier_mod_1274', 'n_nrecevier_mod_1273', 'n_nrecevier_mod_1274NumDims', 'n_nrecevier_mod_1273NumDims'};
            this.ONNXParams = onnxParams;
            this.const_fold_opt__5883 = onnxParams.Learnables.const_fold_opt__5883;
            this.n_nrecevier_mod_1190 = onnxParams.Learnables.n_nrecevier_mod_1190;
            this.n_nrecevier_mod_1192 = onnxParams.Learnables.n_nrecevier_mod_1192;
        end

        function [n_nrecevier_mod_1282, n_nrecevier_mod_1274, n_nrecevier_mod_1273, n_nrecevier_mod_1274NumDims, n_nrecevier_mod_1273NumDims] = predict(this, n_nrecevier_mod_963, Transpose__5103_0)
            if isdlarray(n_nrecevier_mod_963)
                n_nrecevier_mod_963 = stripdims(n_nrecevier_mod_963);
            end
            if isdlarray(Transpose__5103_0)
                Transpose__5103_0 = stripdims(Transpose__5103_0);
            end
            n_nrecevier_mod_963NumDims = 4;
            Transpose__5103_0NumDims = 4;
            onnxParams = this.ONNXParams;
            onnxParams.Learnables.const_fold_opt__5883 = this.const_fold_opt__5883;
            onnxParams.Learnables.n_nrecevier_mod_1190 = this.n_nrecevier_mod_1190;
            onnxParams.Learnables.n_nrecevier_mod_1192 = this.n_nrecevier_mod_1192;
            [n_nrecevier_mod_1282, n_nrecevier_mod_1274, n_nrecevier_mod_1273, n_nrecevier_mod_1282NumDims, n_nrecevier_mod_1274NumDims, n_nrecevier_mod_1273NumDims] = Shape_To_ReshapeFcn(n_nrecevier_mod_963, Transpose__5103_0, n_nrecevier_mod_963NumDims, Transpose__5103_0NumDims, onnxParams, 'Training', false, ...
                'InputDataPermutation', {[4 3 1 2], [4 1 2 3], ['as-is'], ['as-is']}, ...
                'OutputDataPermutation', {[2 3 4 1], ['as-is'], ['as-is'], ['as-is'], ['as-is'], ['as-is']});
            if any(cellfun(@(A)~isnumeric(A) && ~islogical(A), {n_nrecevier_mod_1282, n_nrecevier_mod_1274, n_nrecevier_mod_1273}))
                fprintf('Runtime error in network. At least one output of custom layer ''%s'' is a non-numeric, non-logical value.\n', 'Shape_To_ReshapeLayer1145');
                error(message('nnet_cnn_onnx:onnx:BadCustomLayerRuntimeOutput', 'Shape_To_ReshapeLayer1145'));
            end
            n_nrecevier_mod_1282 = dlarray(single(n_nrecevier_mod_1282), 'SSCB');
            n_nrecevier_mod_1274 = dlarray(single(n_nrecevier_mod_1274), repmat('U', 1, max(2, n_nrecevier_mod_1274NumDims)));
            n_nrecevier_mod_1273 = dlarray(single(n_nrecevier_mod_1273), repmat('U', 1, max(2, n_nrecevier_mod_1273NumDims)));
            n_nrecevier_mod_1274NumDims = dlarray(ones(1,n_nrecevier_mod_1274NumDims,'like',n_nrecevier_mod_1282), 'UU');
            n_nrecevier_mod_1273NumDims = dlarray(ones(1,n_nrecevier_mod_1273NumDims,'like',n_nrecevier_mod_1282), 'UU');
            if ~coder.target('MATLAB')
                n_nrecevier_mod_1282 = extractdata(n_nrecevier_mod_1282);
                n_nrecevier_mod_1274 = extractdata(n_nrecevier_mod_1274);
                n_nrecevier_mod_1273 = extractdata(n_nrecevier_mod_1273);
                n_nrecevier_mod_1274NumDims = extractdata(n_nrecevier_mod_1274NumDims);
                n_nrecevier_mod_1273NumDims = extractdata(n_nrecevier_mod_1273NumDims);
            end
        end

        function [n_nrecevier_mod_1282, n_nrecevier_mod_1274, n_nrecevier_mod_1273, n_nrecevier_mod_1274NumDims, n_nrecevier_mod_1273NumDims] = forward(this, n_nrecevier_mod_963, Transpose__5103_0)
            if isdlarray(n_nrecevier_mod_963)
                n_nrecevier_mod_963 = stripdims(n_nrecevier_mod_963);
            end
            if isdlarray(Transpose__5103_0)
                Transpose__5103_0 = stripdims(Transpose__5103_0);
            end
            n_nrecevier_mod_963NumDims = 4;
            Transpose__5103_0NumDims = 4;
            onnxParams = this.ONNXParams;
            onnxParams.Learnables.const_fold_opt__5883 = this.const_fold_opt__5883;
            onnxParams.Learnables.n_nrecevier_mod_1190 = this.n_nrecevier_mod_1190;
            onnxParams.Learnables.n_nrecevier_mod_1192 = this.n_nrecevier_mod_1192;
            [n_nrecevier_mod_1282, n_nrecevier_mod_1274, n_nrecevier_mod_1273, n_nrecevier_mod_1282NumDims, n_nrecevier_mod_1274NumDims, n_nrecevier_mod_1273NumDims] = Shape_To_ReshapeFcn(n_nrecevier_mod_963, Transpose__5103_0, n_nrecevier_mod_963NumDims, Transpose__5103_0NumDims, onnxParams, 'Training', true, ...
                'InputDataPermutation', {[4 3 1 2], [4 1 2 3], ['as-is'], ['as-is']}, ...
                'OutputDataPermutation', {[2 3 4 1], ['as-is'], ['as-is'], ['as-is'], ['as-is'], ['as-is']});
            if any(cellfun(@(A)~isnumeric(A) && ~islogical(A), {n_nrecevier_mod_1282, n_nrecevier_mod_1274, n_nrecevier_mod_1273}))
                fprintf('Runtime error in network. At least one output of custom layer ''%s'' is a non-numeric, non-logical value.\n', 'Shape_To_ReshapeLayer1145');
                error(message('nnet_cnn_onnx:onnx:BadCustomLayerRuntimeOutput', 'Shape_To_ReshapeLayer1145'));
            end
            n_nrecevier_mod_1282 = dlarray(single(n_nrecevier_mod_1282), 'SSCB');
            n_nrecevier_mod_1274 = dlarray(single(n_nrecevier_mod_1274), repmat('U', 1, max(2, n_nrecevier_mod_1274NumDims)));
            n_nrecevier_mod_1273 = dlarray(single(n_nrecevier_mod_1273), repmat('U', 1, max(2, n_nrecevier_mod_1273NumDims)));
            n_nrecevier_mod_1274NumDims = dlarray(ones(1,n_nrecevier_mod_1274NumDims,'like',n_nrecevier_mod_1282), 'UU');
            n_nrecevier_mod_1273NumDims = dlarray(ones(1,n_nrecevier_mod_1273NumDims,'like',n_nrecevier_mod_1282), 'UU');
            if ~coder.target('MATLAB')
                n_nrecevier_mod_1282 = extractdata(n_nrecevier_mod_1282);
                n_nrecevier_mod_1274 = extractdata(n_nrecevier_mod_1274);
                n_nrecevier_mod_1273 = extractdata(n_nrecevier_mod_1273);
                n_nrecevier_mod_1274NumDims = extractdata(n_nrecevier_mod_1274NumDims);
                n_nrecevier_mod_1273NumDims = extractdata(n_nrecevier_mod_1273NumDims);
            end
        end
    end
end

function [n_nrecevier_mod_1282, n_nrecevier_mod_1274, n_nrecevier_mod_1273, n_nrecevier_mod_1282NumDims, n_nrecevier_mod_1274NumDims, n_nrecevier_mod_1273NumDims, state] = Shape_To_ReshapeFcn(n_nrecevier_mod_963, Transpose__5103_0, n_nrecevier_mod_963NumDims, Transpose__5103_0NumDims, params, varargin)
%SHAPE_TO_RESHAPEFCN Function implementing an imported ONNX network.
%
% THIS FILE WAS AUTO-GENERATED BY importONNXFunction.
% ONNX Operator Set Version: 18
%
% Variable names in this function are taken from the original ONNX file.
%
% [N_NRECEVIER_MOD_1282, N_NRECEVIER_MOD_1274, N_NRECEVIER_MOD_1273] = Shape_To_ReshapeFcn(N_NRECEVIER_MOD_963, TRANSPOSE__5103_0, PARAMS)
%			- Evaluates the imported ONNX network SHAPE_TO_RESHAPEFCN with input(s)
%			N_NRECEVIER_MOD_963, TRANSPOSE__5103_0 and the imported network parameters in PARAMS. Returns
%			network output(s) in N_NRECEVIER_MOD_1282, N_NRECEVIER_MOD_1274, N_NRECEVIER_MOD_1273.
%
% [N_NRECEVIER_MOD_1282, N_NRECEVIER_MOD_1274, N_NRECEVIER_MOD_1273, STATE] = Shape_To_ReshapeFcn(N_NRECEVIER_MOD_963, TRANSPOSE__5103_0, PARAMS)
%			- Additionally returns state variables in STATE. When training,
%			use this form and set TRAINING to true.
%
% [__] = Shape_To_ReshapeFcn(N_NRECEVIER_MOD_963, TRANSPOSE__5103_0, PARAMS, 'NAME1', VAL1, 'NAME2', VAL2, ...)
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
% N_NRECEVIER_MOD_963, TRANSPOSE__5103_0
%			- Input(s) to the ONNX network.
%			  The input size(s) expected by the ONNX file are:
%				  N_NRECEVIER_MOD_963:		[Unknown, Unknown, Unknown, Unknown]				Type: FLOAT
%				  TRANSPOSE__5103_0:		[Unknown, Unknown, Unknown, Unknown]				Type: FLOAT
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
% N_NRECEVIER_MOD_1282, N_NRECEVIER_MOD_1274, N_NRECEVIER_MOD_1273
%			- Output(s) of the ONNX network.
%			  Without permutation, the size(s) of the outputs are:
%				  N_NRECEVIER_MOD_1282:		[Unknown, Unknown, Unknown, Unknown]				Type: FLOAT
%				  N_NRECEVIER_MOD_1274:		[1, 1]				Type: FLOAT
%				  N_NRECEVIER_MOD_1273:		[1, 1]				Type: FLOAT
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
[n_nrecevier_mod_963, Transpose__5103_0, Training, outputDataPerms, anyDlarrayInputs] = preprocessInput(n_nrecevier_mod_963, Transpose__5103_0, params, varargin{:});
% Put all variables into a single struct to implement dynamic scoping:
[Vars, NumDims] = packageVariables(params, {'n_nrecevier_mod_963', 'Transpose__5103_0'}, {n_nrecevier_mod_963, Transpose__5103_0}, [n_nrecevier_mod_963NumDims Transpose__5103_0NumDims]);
% Call the top-level graph function:
[n_nrecevier_mod_1282, n_nrecevier_mod_1274, n_nrecevier_mod_1273, n_nrecevier_mod_1282NumDims, n_nrecevier_mod_1274NumDims, n_nrecevier_mod_1273NumDims, state] = Shape_To_ReshapeGraph1135(n_nrecevier_mod_963, Transpose__5103_0, NumDims.n_nrecevier_mod_963, NumDims.Transpose__5103_0, Vars, NumDims, Training, params.State);
% Postprocess the output data
[n_nrecevier_mod_1282, n_nrecevier_mod_1274, n_nrecevier_mod_1273] = postprocessOutput(n_nrecevier_mod_1282, n_nrecevier_mod_1274, n_nrecevier_mod_1273, outputDataPerms, anyDlarrayInputs, Training, varargin{:});
end

function [n_nrecevier_mod_1282, n_nrecevier_mod_1274, n_nrecevier_mod_1273, n_nrecevier_mod_1282NumDims1142, n_nrecevier_mod_1274NumDims1143, n_nrecevier_mod_1273NumDims1144, state] = Shape_To_ReshapeGraph1135(n_nrecevier_mod_963, Transpose__5103_0, n_nrecevier_mod_963NumDims1140, Transpose__5103_0NumDims1141, Vars, NumDims, Training, state)
% Function implementing the graph 'Shape_To_ReshapeGraph1135'
% Update Vars and NumDims from the graph's formal input parameters. Note that state variables are already in Vars.
Vars.n_nrecevier_mod_963 = n_nrecevier_mod_963;
NumDims.n_nrecevier_mod_963 = n_nrecevier_mod_963NumDims1140;
Vars.Transpose__5103_0 = Transpose__5103_0;
NumDims.Transpose__5103_0 = Transpose__5103_0NumDims1141;

% Execute the operators:
% Shape:
[Vars.Shape__5549_0, NumDims.Shape__5549_0] = onnxShape(Vars.n_nrecevier_mod_963, NumDims.n_nrecevier_mod_963, 0, NumDims.n_nrecevier_mod_963+1);

% Gather:
[Vars.n_nrecevier_mod_1200, NumDims.n_nrecevier_mod_1200] = onnxGather(Vars.Shape__5549_0, Vars.Const__5379, 0, NumDims.Shape__5549_0, NumDims.Const__5379);

% Cast:
if islogical(Vars.n_nrecevier_mod_1200)
    Vars.n_nrecevier_mod_1200 = single(Vars.n_nrecevier_mod_1200);
end
Vars.n_nrecevier_mod_1201 = cast(int32(extractdata(Vars.n_nrecevier_mod_1200)), 'like', Vars.n_nrecevier_mod_1200);
NumDims.n_nrecevier_mod_1201 = NumDims.n_nrecevier_mod_1200;

% Slice:
[Indices, NumDims.n_nrecevier_mod_1221] = prepareSliceArgs(Vars.n_nrecevier_mod_1201, Vars.const__2055, Vars.const_ends__3053, Vars.const_axes__4255, '', NumDims.n_nrecevier_mod_1201);
Vars.n_nrecevier_mod_1221 = subsref(Vars.n_nrecevier_mod_1201, Indices);

% Squeeze:
[Vars.n_nrecevier_mod_1220, NumDims.n_nrecevier_mod_1220] = onnxSqueeze(Vars.n_nrecevier_mod_1221, Vars.const_axes__4255, NumDims.n_nrecevier_mod_1221);

% Div:
Vars.n_nrecevier_mod_1211 = fix(Vars.n_nrecevier_mod_1220 ./ Vars.n_nrecevier_mod_1210);
NumDims.n_nrecevier_mod_1211 = max(NumDims.n_nrecevier_mod_1220, NumDims.n_nrecevier_mod_1210);

% Unsqueeze:
[shape, NumDims.n_nrecevier_mod_1194] = prepareUnsqueezeArgs(Vars.n_nrecevier_mod_1211, Vars.const_axes__4255, NumDims.n_nrecevier_mod_1211);
Vars.n_nrecevier_mod_1194 = reshape(Vars.n_nrecevier_mod_1211, shape);

% Concat:
[Vars.n_nrecevier_mod_1193, NumDims.n_nrecevier_mod_1193] = onnxConcat(0, {Vars.const_fold_opt__5935, Vars.const_fold_opt__5935, Vars.const_fold_opt__5935, Vars.const_fold_opt__6005, Vars.n_nrecevier_mod_1194}, [NumDims.const_fold_opt__5935, NumDims.const_fold_opt__5935, NumDims.const_fold_opt__5935, NumDims.const_fold_opt__6005, NumDims.n_nrecevier_mod_1194]);

% Cast:
if islogical(Vars.n_nrecevier_mod_1193)
    Vars.n_nrecevier_mod_1193 = single(Vars.n_nrecevier_mod_1193);
end
Vars.n_nrecevier_mod_1196 = cast(int64(extractdata(Vars.n_nrecevier_mod_1193)), 'like', Vars.n_nrecevier_mod_1193);
NumDims.n_nrecevier_mod_1196 = NumDims.n_nrecevier_mod_1193;

% Reshape:
[shape, NumDims.n_nrecevier_mod_1195] = prepareReshapeArgs(Vars.n_nrecevier_mod_1192, Vars.n_nrecevier_mod_1196, NumDims.n_nrecevier_mod_1192, 0);
Vars.n_nrecevier_mod_1195 = reshape(Vars.n_nrecevier_mod_1192, shape{:});

% Reshape:
[shape, NumDims.n_nrecevier_mod_1191] = prepareReshapeArgs(Vars.n_nrecevier_mod_1190, Vars.n_nrecevier_mod_1196, NumDims.n_nrecevier_mod_1190, 0);
Vars.n_nrecevier_mod_1191 = reshape(Vars.n_nrecevier_mod_1190, shape{:});

% Slice:
[Indices, NumDims.n_nrecevier_mod_1219] = prepareSliceArgs(Vars.n_nrecevier_mod_1201, Vars.const_starts__1983, Vars.const__738, Vars.const_axes__4255, '', NumDims.n_nrecevier_mod_1201);
Vars.n_nrecevier_mod_1219 = subsref(Vars.n_nrecevier_mod_1201, Indices);

% Slice:
[Indices, NumDims.n_nrecevier_mod_1218] = prepareSliceArgs(Vars.n_nrecevier_mod_1201, Vars.const_starts__1988, Vars.const_starts__1983, Vars.const_axes__4255, '', NumDims.n_nrecevier_mod_1201);
Vars.n_nrecevier_mod_1218 = subsref(Vars.n_nrecevier_mod_1201, Indices);

% Slice:
[Indices, NumDims.n_nrecevier_mod_1217] = prepareSliceArgs(Vars.n_nrecevier_mod_1201, Vars.const_axes__4255, Vars.const_starts__1988, Vars.const_axes__4255, '', NumDims.n_nrecevier_mod_1201);
Vars.n_nrecevier_mod_1217 = subsref(Vars.n_nrecevier_mod_1201, Indices);

% Concat:
[Vars.n_nrecevier_mod_1216, NumDims.n_nrecevier_mod_1216] = onnxConcat(0, {Vars.n_nrecevier_mod_1217, Vars.n_nrecevier_mod_1218, Vars.n_nrecevier_mod_1219, Vars.const_fold_opt__6005, Vars.n_nrecevier_mod_1194}, [NumDims.n_nrecevier_mod_1217, NumDims.n_nrecevier_mod_1218, NumDims.n_nrecevier_mod_1219, NumDims.const_fold_opt__6005, NumDims.n_nrecevier_mod_1194]);

% Cast:
if islogical(Vars.n_nrecevier_mod_1216)
    Vars.n_nrecevier_mod_1216 = single(Vars.n_nrecevier_mod_1216);
end
Vars.n_nrecevier_mod_1199 = cast(int64(extractdata(Vars.n_nrecevier_mod_1216)), 'like', Vars.n_nrecevier_mod_1216);
NumDims.n_nrecevier_mod_1199 = NumDims.n_nrecevier_mod_1216;

% Reshape:
[shape, NumDims.n_nrecevier_mod_1189] = prepareReshapeArgs(Vars.Transpose__5103_0, Vars.n_nrecevier_mod_1199, NumDims.Transpose__5103_0, 0);
Vars.n_nrecevier_mod_1189 = reshape(Vars.Transpose__5103_0, shape{:});

% ReduceMean:
dims = prepareReduceArgs(Vars.const_fold_opt__5883, NumDims.n_nrecevier_mod_1189);
Vars.n_nrecevier_mod_1214 = mean(Vars.n_nrecevier_mod_1189, dims);
NumDims.n_nrecevier_mod_1214 = NumDims.n_nrecevier_mod_1189;

% Sub:
Vars.n_nrecevier_mod_1212 = Vars.n_nrecevier_mod_1189 - Vars.n_nrecevier_mod_1214;
NumDims.n_nrecevier_mod_1212 = max(NumDims.n_nrecevier_mod_1189, NumDims.n_nrecevier_mod_1214);

% Mul:
Vars.n_nrecevier_mod_1213 = Vars.n_nrecevier_mod_1212 .* Vars.n_nrecevier_mod_1212;
NumDims.n_nrecevier_mod_1213 = max(NumDims.n_nrecevier_mod_1212, NumDims.n_nrecevier_mod_1212);

% ReduceMean:
dims = prepareReduceArgs(Vars.const_fold_opt__5883, NumDims.n_nrecevier_mod_1213);
Vars.n_nrecevier_mod_1215 = mean(Vars.n_nrecevier_mod_1213, dims);
NumDims.n_nrecevier_mod_1215 = NumDims.n_nrecevier_mod_1213;

% Add:
Vars.n_nrecevier_mod_1204 = Vars.n_nrecevier_mod_1215 + Vars.n_nrecevier_mod_146;
NumDims.n_nrecevier_mod_1204 = max(NumDims.n_nrecevier_mod_1215, NumDims.n_nrecevier_mod_146);

% Sqrt:
Vars.n_nrecevier_mod_1202 = sqrt(Vars.n_nrecevier_mod_1204);
NumDims.n_nrecevier_mod_1202 = NumDims.n_nrecevier_mod_1204;

% Reciprocal:
Vars.n_nrecevier_mod_1203 = 1./(Vars.n_nrecevier_mod_1202);
NumDims.n_nrecevier_mod_1203 = NumDims.n_nrecevier_mod_1202;

% Mul:
Vars.n_nrecevier_mod_1206 = Vars.n_nrecevier_mod_1203 .* Vars.n_nrecevier_mod_1191;
NumDims.n_nrecevier_mod_1206 = max(NumDims.n_nrecevier_mod_1203, NumDims.n_nrecevier_mod_1191);

% Mul:
Vars.n_nrecevier_mod_1208 = Vars.n_nrecevier_mod_1214 .* Vars.n_nrecevier_mod_1206;
NumDims.n_nrecevier_mod_1208 = max(NumDims.n_nrecevier_mod_1214, NumDims.n_nrecevier_mod_1206);

% Sub:
Vars.n_nrecevier_mod_1209 = Vars.n_nrecevier_mod_1195 - Vars.n_nrecevier_mod_1208;
NumDims.n_nrecevier_mod_1209 = max(NumDims.n_nrecevier_mod_1195, NumDims.n_nrecevier_mod_1208);

% Mul:
Vars.n_nrecevier_mod_1207 = Vars.n_nrecevier_mod_1189 .* Vars.n_nrecevier_mod_1206;
NumDims.n_nrecevier_mod_1207 = max(NumDims.n_nrecevier_mod_1189, NumDims.n_nrecevier_mod_1206);

% Add:
Vars.n_nrecevier_mod_1205 = Vars.n_nrecevier_mod_1207 + Vars.n_nrecevier_mod_1209;
NumDims.n_nrecevier_mod_1205 = max(NumDims.n_nrecevier_mod_1207, NumDims.n_nrecevier_mod_1209);

% Cast:
if islogical(Vars.n_nrecevier_mod_1201)
    Vars.n_nrecevier_mod_1201 = single(Vars.n_nrecevier_mod_1201);
end
Vars.n_nrecevier_mod_1198 = cast(int64(extractdata(Vars.n_nrecevier_mod_1201)), 'like', Vars.n_nrecevier_mod_1201);
NumDims.n_nrecevier_mod_1198 = NumDims.n_nrecevier_mod_1201;

% Reshape:
[shape, NumDims.n_nrecevier_mod_1197] = prepareReshapeArgs(Vars.n_nrecevier_mod_1205, Vars.n_nrecevier_mod_1198, NumDims.n_nrecevier_mod_1205, 0);
Vars.n_nrecevier_mod_1197 = reshape(Vars.n_nrecevier_mod_1205, shape{:});

% Relu:
Vars.n_nrecevier_mod_1186 = relu(Vars.n_nrecevier_mod_1197);
NumDims.n_nrecevier_mod_1186 = NumDims.n_nrecevier_mod_1197;

% Shape:
[Vars.n_nrecevier_mod_1283, NumDims.n_nrecevier_mod_1283] = onnxShape(Vars.n_nrecevier_mod_1186, NumDims.n_nrecevier_mod_1186, 0, NumDims.n_nrecevier_mod_1186+1);

% PLACEHOLDER FUNCTION FOR UNSUPPORTED OPERATOR (Size):
[Vars.n_nrecevier_mod_1299, NumDims.n_nrecevier_mod_1299] = PLACEHOLDER(Vars.n_nrecevier_mod_1283);

% Shape:
[Vars.n_nrecevier_mod_1280, NumDims.n_nrecevier_mod_1280] = onnxShape(Vars.n_nrecevier_mod_1186, NumDims.n_nrecevier_mod_1186, 0, NumDims.n_nrecevier_mod_1186+1);

% Cast:
if islogical(Vars.n_nrecevier_mod_1280)
    Vars.n_nrecevier_mod_1280 = single(Vars.n_nrecevier_mod_1280);
end
Vars.n_nrecevier_mod_1281 = cast(int32(extractdata(Vars.n_nrecevier_mod_1280)), 'like', Vars.n_nrecevier_mod_1280);
NumDims.n_nrecevier_mod_1281 = NumDims.n_nrecevier_mod_1280;

% Slice:
[Indices, NumDims.n_nrecevier_mod_1331] = prepareSliceArgs(Vars.n_nrecevier_mod_1281, Vars.const_starts__1983, Vars.const__738, Vars.const_axes__4255, '', NumDims.n_nrecevier_mod_1281);
Vars.n_nrecevier_mod_1331 = subsref(Vars.n_nrecevier_mod_1281, Indices);

% Slice:
[Indices, NumDims.n_nrecevier_mod_1330] = prepareSliceArgs(Vars.n_nrecevier_mod_1281, Vars.const_starts__1988, Vars.const_starts__1983, Vars.const_axes__4255, '', NumDims.n_nrecevier_mod_1281);
Vars.n_nrecevier_mod_1330 = subsref(Vars.n_nrecevier_mod_1281, Indices);

% Concat:
[Vars.n_nrecevier_mod_1329, NumDims.n_nrecevier_mod_1329] = onnxConcat(0, {Vars.n_nrecevier_mod_1330, Vars.n_nrecevier_mod_1331}, [NumDims.n_nrecevier_mod_1330, NumDims.n_nrecevier_mod_1331]);

% Add:
Vars.n_nrecevier_mod_1310 = Vars.n_nrecevier_mod_1329 + Vars.n_nrecevier_mod_938;
NumDims.n_nrecevier_mod_1310 = max(NumDims.n_nrecevier_mod_1329, NumDims.n_nrecevier_mod_938);

% Div:
Vars.Div__2834_0 = fix(Vars.n_nrecevier_mod_1310 ./ Vars.n_nrecevier_mod_937);
NumDims.Div__2834_0 = max(NumDims.n_nrecevier_mod_1310, NumDims.n_nrecevier_mod_937);

% Mul:
Vars.Mul__2835_0 = Vars.Div__2834_0 .* Vars.n_nrecevier_mod_937;
NumDims.Mul__2835_0 = max(NumDims.Div__2834_0, NumDims.n_nrecevier_mod_937);

% Sub:
Vars.n_nrecevier_mod_1317 = Vars.n_nrecevier_mod_1310 - Vars.Mul__2835_0;
NumDims.n_nrecevier_mod_1317 = max(NumDims.n_nrecevier_mod_1310, NumDims.Mul__2835_0);

% Sub:
Vars.n_nrecevier_mod_1328 = Vars.n_nrecevier_mod_937 - Vars.n_nrecevier_mod_1317;
NumDims.n_nrecevier_mod_1328 = max(NumDims.n_nrecevier_mod_937, NumDims.n_nrecevier_mod_1317);

% Div:
Vars.Div__2836_0 = fix(Vars.n_nrecevier_mod_1328 ./ Vars.n_nrecevier_mod_937);
NumDims.Div__2836_0 = max(NumDims.n_nrecevier_mod_1328, NumDims.n_nrecevier_mod_937);

% Mul:
Vars.Mul__2837_0 = Vars.Div__2836_0 .* Vars.n_nrecevier_mod_937;
NumDims.Mul__2837_0 = max(NumDims.Div__2836_0, NumDims.n_nrecevier_mod_937);

% Sub:
Vars.n_nrecevier_mod_1318 = Vars.n_nrecevier_mod_1328 - Vars.Mul__2837_0;
NumDims.n_nrecevier_mod_1318 = max(NumDims.n_nrecevier_mod_1328, NumDims.Mul__2837_0);

% Slice:
[Indices, NumDims.n_nrecevier_mod_1327] = prepareSliceArgs(Vars.n_nrecevier_mod_1318, Vars.const_starts__1988, Vars.const_starts__1983, Vars.const_axes__4255, '', NumDims.n_nrecevier_mod_1318);
Vars.n_nrecevier_mod_1327 = subsref(Vars.n_nrecevier_mod_1318, Indices);

% Concat:
[Vars.n_nrecevier_mod_1313, NumDims.n_nrecevier_mod_1313] = onnxConcat(0, {Vars.const_fold_opt__5837, Vars.n_nrecevier_mod_1327}, [NumDims.const_fold_opt__5837, NumDims.n_nrecevier_mod_1327]);

% Unsqueeze:
[shape, NumDims.n_nrecevier_mod_1316] = prepareUnsqueezeArgs(Vars.n_nrecevier_mod_1313, Vars.const_axes__4255, NumDims.n_nrecevier_mod_1313);
Vars.n_nrecevier_mod_1316 = reshape(Vars.n_nrecevier_mod_1313, shape);

% Slice:
[Indices, NumDims.n_nrecevier_mod_1326] = prepareSliceArgs(Vars.n_nrecevier_mod_1318, Vars.const_axes__4255, Vars.const_starts__1988, Vars.const_axes__4255, '', NumDims.n_nrecevier_mod_1318);
Vars.n_nrecevier_mod_1326 = subsref(Vars.n_nrecevier_mod_1318, Indices);

% Concat:
[Vars.n_nrecevier_mod_1312, NumDims.n_nrecevier_mod_1312] = onnxConcat(0, {Vars.const_fold_opt__5837, Vars.n_nrecevier_mod_1326}, [NumDims.const_fold_opt__5837, NumDims.n_nrecevier_mod_1326]);

% Unsqueeze:
[shape, NumDims.n_nrecevier_mod_1315] = prepareUnsqueezeArgs(Vars.n_nrecevier_mod_1312, Vars.const_axes__4255, NumDims.n_nrecevier_mod_1312);
Vars.n_nrecevier_mod_1315 = reshape(Vars.n_nrecevier_mod_1312, shape);

% Concat:
[Vars.n_nrecevier_mod_1314, NumDims.n_nrecevier_mod_1314] = onnxConcat(0, {Vars.n_nrecevier_mod_1315, Vars.n_nrecevier_mod_1316}, [NumDims.n_nrecevier_mod_1315, NumDims.n_nrecevier_mod_1316]);

% Cast:
if islogical(Vars.n_nrecevier_mod_1314)
    Vars.n_nrecevier_mod_1314 = single(Vars.n_nrecevier_mod_1314);
end
Vars.n_nrecevier_mod_1259 = cast(int64(extractdata(Vars.n_nrecevier_mod_1314)), 'like', Vars.n_nrecevier_mod_1314);
NumDims.n_nrecevier_mod_1259 = NumDims.n_nrecevier_mod_1314;

% Transpose:
[perm, NumDims.n_nrecevier_mod_1278] = prepareTransposeArgs('', NumDims.n_nrecevier_mod_1259);
if ~isempty(perm)
    Vars.n_nrecevier_mod_1278 = permute(Vars.n_nrecevier_mod_1259, perm);
end

% Slice:
[Indices, NumDims.n_nrecevier_mod_1272] = prepareSliceArgs(Vars.n_nrecevier_mod_1278, Vars.const__1051, Vars.const__1888, '', '', NumDims.n_nrecevier_mod_1278);
Vars.n_nrecevier_mod_1272 = subsref(Vars.n_nrecevier_mod_1278, Indices);

% Squeeze:
[Vars.n_nrecevier_mod_1274, NumDims.n_nrecevier_mod_1274] = onnxSqueeze(Vars.n_nrecevier_mod_1272, Vars.const_axes__4255, NumDims.n_nrecevier_mod_1272);

% Slice:
[Indices, NumDims.n_nrecevier_mod_1271] = prepareSliceArgs(Vars.n_nrecevier_mod_1278, Vars.const__773, Vars.const__774, '', '', NumDims.n_nrecevier_mod_1278);
Vars.n_nrecevier_mod_1271 = subsref(Vars.n_nrecevier_mod_1278, Indices);

% Squeeze:
[Vars.n_nrecevier_mod_1273, NumDims.n_nrecevier_mod_1273] = onnxSqueeze(Vars.n_nrecevier_mod_1271, Vars.const_axes__4255, NumDims.n_nrecevier_mod_1271);

% Add:
Vars.n_nrecevier_mod_1311 = Vars.n_nrecevier_mod_937 + Vars.n_nrecevier_mod_1318;
NumDims.n_nrecevier_mod_1311 = max(NumDims.n_nrecevier_mod_937, NumDims.n_nrecevier_mod_1318);

% Slice:
[Indices, NumDims.n_nrecevier_mod_1325] = prepareSliceArgs(Vars.n_nrecevier_mod_1311, Vars.const_starts__1988, Vars.const_starts__1983, Vars.const_axes__4255, '', NumDims.n_nrecevier_mod_1311);
Vars.n_nrecevier_mod_1325 = subsref(Vars.n_nrecevier_mod_1311, Indices);

% Concat:
[Vars.n_nrecevier_mod_1320, NumDims.n_nrecevier_mod_1320] = onnxConcat(0, {Vars.const_fold_opt__5913, Vars.n_nrecevier_mod_1325}, [NumDims.const_fold_opt__5913, NumDims.n_nrecevier_mod_1325]);

% Unsqueeze:
[shape, NumDims.n_nrecevier_mod_1323] = prepareUnsqueezeArgs(Vars.n_nrecevier_mod_1320, Vars.const_axes__4255, NumDims.n_nrecevier_mod_1320);
Vars.n_nrecevier_mod_1323 = reshape(Vars.n_nrecevier_mod_1320, shape);

% Slice:
[Indices, NumDims.n_nrecevier_mod_1324] = prepareSliceArgs(Vars.n_nrecevier_mod_1311, Vars.const_axes__4255, Vars.const_starts__1988, Vars.const_axes__4255, '', NumDims.n_nrecevier_mod_1311);
Vars.n_nrecevier_mod_1324 = subsref(Vars.n_nrecevier_mod_1311, Indices);

% Concat:
[Vars.n_nrecevier_mod_1319, NumDims.n_nrecevier_mod_1319] = onnxConcat(0, {Vars.const_fold_opt__5912, Vars.n_nrecevier_mod_1324}, [NumDims.const_fold_opt__5912, NumDims.n_nrecevier_mod_1324]);

% Unsqueeze:
[shape, NumDims.n_nrecevier_mod_1322] = prepareUnsqueezeArgs(Vars.n_nrecevier_mod_1319, Vars.const_axes__4255, NumDims.n_nrecevier_mod_1319);
Vars.n_nrecevier_mod_1322 = reshape(Vars.n_nrecevier_mod_1319, shape);

% Concat:
[Vars.n_nrecevier_mod_1321, NumDims.n_nrecevier_mod_1321] = onnxConcat(0, {Vars.n_nrecevier_mod_1322, Vars.n_nrecevier_mod_1323}, [NumDims.n_nrecevier_mod_1322, NumDims.n_nrecevier_mod_1323]);

% Cast:
if islogical(Vars.n_nrecevier_mod_1321)
    Vars.n_nrecevier_mod_1321 = single(Vars.n_nrecevier_mod_1321);
end
Vars.n_nrecevier_mod_1284 = cast(int64(extractdata(Vars.n_nrecevier_mod_1321)), 'like', Vars.n_nrecevier_mod_1321);
NumDims.n_nrecevier_mod_1284 = NumDims.n_nrecevier_mod_1321;

% Shape:
[Vars.n_nrecevier_mod_1297, NumDims.n_nrecevier_mod_1297] = onnxShape(Vars.n_nrecevier_mod_1284, NumDims.n_nrecevier_mod_1284, 0, NumDims.n_nrecevier_mod_1284+1);

% Slice:
[Indices, NumDims.n_nrecevier_mod_1300] = prepareSliceArgs(Vars.n_nrecevier_mod_1297, Vars.const_axes__4255, Vars.const_starts__1988, '', '', NumDims.n_nrecevier_mod_1297);
Vars.n_nrecevier_mod_1300 = subsref(Vars.n_nrecevier_mod_1297, Indices);

% Sub:
Vars.n_nrecevier_mod_1303 = Vars.n_nrecevier_mod_1299 - Vars.n_nrecevier_mod_1300;
NumDims.n_nrecevier_mod_1303 = max(NumDims.n_nrecevier_mod_1299, NumDims.n_nrecevier_mod_1300);

% Sub:
Vars.n_nrecevier_mod_1304 = Vars.n_nrecevier_mod_1303 - Vars.const_starts__1988;
NumDims.n_nrecevier_mod_1304 = max(NumDims.n_nrecevier_mod_1303, NumDims.const_starts__1988);

% Mul:
Vars.n_nrecevier_mod_1289 = Vars.const__1231 .* Vars.n_nrecevier_mod_1304;
NumDims.n_nrecevier_mod_1289 = max(NumDims.const__1231, NumDims.n_nrecevier_mod_1304);

% Pad:
[Vars.n_nrecevier_mod_1290, NumDims.n_nrecevier_mod_1290] = onnxPad(Vars.n_nrecevier_mod_1284, Vars.const__1230, 0, 'constant', dlarray([0:NumDims.n_nrecevier_mod_1284]'), NumDims.n_nrecevier_mod_1284);

% Pad:
[Vars.n_nrecevier_mod_1291, NumDims.n_nrecevier_mod_1291] = onnxPad(Vars.n_nrecevier_mod_1290, Vars.n_nrecevier_mod_1289, 0, 'constant', dlarray([0:NumDims.n_nrecevier_mod_1290]'), NumDims.n_nrecevier_mod_1290);

% Transpose:
[perm, NumDims.n_nrecevier_mod_1305] = prepareTransposeArgs('', NumDims.n_nrecevier_mod_1291);
if ~isempty(perm)
    Vars.n_nrecevier_mod_1305 = permute(Vars.n_nrecevier_mod_1291, perm);
end

% Reshape:
[shape, NumDims.n_nrecevier_mod_1293] = prepareReshapeArgs(Vars.n_nrecevier_mod_1305, Vars.const__2055, NumDims.n_nrecevier_mod_1305, 0);
Vars.n_nrecevier_mod_1293 = reshape(Vars.n_nrecevier_mod_1305, shape{:});

% Pad:
[Vars.n_nrecevier_mod_1292, NumDims.n_nrecevier_mod_1292] = onnxPad(Vars.n_nrecevier_mod_1186, Vars.n_nrecevier_mod_1293, 0, 'constant', dlarray([0:NumDims.n_nrecevier_mod_1186]'), NumDims.n_nrecevier_mod_1186);

% Shape:
[Vars.n_nrecevier_mod_1298, NumDims.n_nrecevier_mod_1298] = onnxShape(Vars.n_nrecevier_mod_1292, NumDims.n_nrecevier_mod_1292, 0, NumDims.n_nrecevier_mod_1292+1);

% Slice:
[Indices, NumDims.n_nrecevier_mod_1302] = prepareSliceArgs(Vars.n_nrecevier_mod_1298, Vars.const__738, Vars.const__664, '', '', NumDims.n_nrecevier_mod_1298);
Vars.n_nrecevier_mod_1302 = subsref(Vars.n_nrecevier_mod_1298, Indices);

% Slice:
[Indices, NumDims.n_nrecevier_mod_1301] = prepareSliceArgs(Vars.n_nrecevier_mod_1298, Vars.const_starts__1988, Vars.const__738, '', '', NumDims.n_nrecevier_mod_1298);
Vars.n_nrecevier_mod_1301 = subsref(Vars.n_nrecevier_mod_1298, Indices);

% Div:
Vars.n_nrecevier_mod_1288 = fix(Vars.n_nrecevier_mod_1301 ./ Vars.const__739);
NumDims.n_nrecevier_mod_1288 = max(NumDims.n_nrecevier_mod_1301, NumDims.const__739);

% Concat:
[Vars.n_nrecevier_mod_1287, NumDims.n_nrecevier_mod_1287] = onnxConcat(0, {Vars.const__2055, Vars.n_nrecevier_mod_1288, Vars.n_nrecevier_mod_1302}, [NumDims.const__2055, NumDims.n_nrecevier_mod_1288, NumDims.n_nrecevier_mod_1302]);

% Concat:
[Vars.n_nrecevier_mod_1285, NumDims.n_nrecevier_mod_1285] = onnxConcat(0, {Vars.n_nrecevier_mod_1288, Vars.const__739}, [NumDims.n_nrecevier_mod_1288, NumDims.const__739]);

% Reshape:
[shape, NumDims.n_nrecevier_mod_1294] = prepareReshapeArgs(Vars.n_nrecevier_mod_1285, Vars.const__1228, NumDims.n_nrecevier_mod_1285, 0);
Vars.n_nrecevier_mod_1294 = reshape(Vars.n_nrecevier_mod_1285, shape{:});

% Transpose:
[perm, NumDims.n_nrecevier_mod_1306] = prepareTransposeArgs('', NumDims.n_nrecevier_mod_1294);
if ~isempty(perm)
    Vars.n_nrecevier_mod_1306 = permute(Vars.n_nrecevier_mod_1294, perm);
end

% Reshape:
[shape, NumDims.n_nrecevier_mod_1295] = prepareReshapeArgs(Vars.n_nrecevier_mod_1306, Vars.const__2055, NumDims.n_nrecevier_mod_1306, 0);
Vars.n_nrecevier_mod_1295 = reshape(Vars.n_nrecevier_mod_1306, shape{:});

% Concat:
[Vars.n_nrecevier_mod_1286, NumDims.n_nrecevier_mod_1286] = onnxConcat(0, {Vars.const__2055, Vars.n_nrecevier_mod_1295, Vars.n_nrecevier_mod_1302}, [NumDims.const__2055, NumDims.n_nrecevier_mod_1295, NumDims.n_nrecevier_mod_1302]);

% Reshape:
[shape, NumDims.n_nrecevier_mod_1296] = prepareReshapeArgs(Vars.n_nrecevier_mod_1292, Vars.n_nrecevier_mod_1286, NumDims.n_nrecevier_mod_1292, 0);
Vars.n_nrecevier_mod_1296 = reshape(Vars.n_nrecevier_mod_1292, shape{:});

% Transpose:
[perm, NumDims.n_nrecevier_mod_1307] = prepareTransposeArgs(Vars.TransposePerm1139, NumDims.n_nrecevier_mod_1296);
if ~isempty(perm)
    Vars.n_nrecevier_mod_1307 = permute(Vars.n_nrecevier_mod_1296, perm);
end

% Reshape:
[shape, NumDims.n_nrecevier_mod_1282] = prepareReshapeArgs(Vars.n_nrecevier_mod_1307, Vars.n_nrecevier_mod_1287, NumDims.n_nrecevier_mod_1307, 0);
Vars.n_nrecevier_mod_1282 = reshape(Vars.n_nrecevier_mod_1307, shape{:});

% Set graph output arguments from Vars and NumDims:
n_nrecevier_mod_1282 = Vars.n_nrecevier_mod_1282;
n_nrecevier_mod_1282NumDims1142 = NumDims.n_nrecevier_mod_1282;
n_nrecevier_mod_1274 = Vars.n_nrecevier_mod_1274;
n_nrecevier_mod_1274NumDims1143 = NumDims.n_nrecevier_mod_1274;
n_nrecevier_mod_1273 = Vars.n_nrecevier_mod_1273;
n_nrecevier_mod_1273NumDims1144 = NumDims.n_nrecevier_mod_1273;
% Set output state from Vars:
state = updateStruct(state, Vars);
end

function [inputDataPerms, outputDataPerms, Training] = parseInputs(n_nrecevier_mod_963, Transpose__5103_0, numDataOutputs, params, varargin)
% Function to validate inputs to Shape_To_ReshapeFcn:
p = inputParser;
isValidArrayInput = @(x)isnumeric(x) || isstring(x);
isValidONNXParameters = @(x)isa(x, 'ONNXParameters');
addRequired(p, 'n_nrecevier_mod_963', isValidArrayInput);
addRequired(p, 'Transpose__5103_0', isValidArrayInput);
addRequired(p, 'params', isValidONNXParameters);
addParameter(p, 'InputDataPermutation', 'auto');
addParameter(p, 'OutputDataPermutation', 'auto');
addParameter(p, 'Training', false);
parse(p, n_nrecevier_mod_963, Transpose__5103_0, params, varargin{:});
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

function [n_nrecevier_mod_963, Transpose__5103_0, Training, outputDataPerms, anyDlarrayInputs] = preprocessInput(n_nrecevier_mod_963, Transpose__5103_0, params, varargin)
% Parse input arguments
[inputDataPerms, outputDataPerms, Training] = parseInputs(n_nrecevier_mod_963, Transpose__5103_0, 3, params, varargin{:});
anyDlarrayInputs = any(cellfun(@(x)isa(x, 'dlarray'), {n_nrecevier_mod_963, Transpose__5103_0}));
% Make the input variables into unlabelled dlarrays:
n_nrecevier_mod_963 = makeUnlabeledDlarray(n_nrecevier_mod_963);
Transpose__5103_0 = makeUnlabeledDlarray(Transpose__5103_0);
% Permute inputs if requested:
n_nrecevier_mod_963 = permuteInputVar(n_nrecevier_mod_963, inputDataPerms{1}, 4);
Transpose__5103_0 = permuteInputVar(Transpose__5103_0, inputDataPerms{2}, 4);
end

function [n_nrecevier_mod_1282, n_nrecevier_mod_1274, n_nrecevier_mod_1273] = postprocessOutput(n_nrecevier_mod_1282, n_nrecevier_mod_1274, n_nrecevier_mod_1273, outputDataPerms, anyDlarrayInputs, Training, varargin)
% Set output type:
if ~anyDlarrayInputs && ~Training
    if isdlarray(n_nrecevier_mod_1282)
        n_nrecevier_mod_1282 = extractdata(n_nrecevier_mod_1282);
    end
    if isdlarray(n_nrecevier_mod_1274)
        n_nrecevier_mod_1274 = extractdata(n_nrecevier_mod_1274);
    end
    if isdlarray(n_nrecevier_mod_1273)
        n_nrecevier_mod_1273 = extractdata(n_nrecevier_mod_1273);
    end
end
% Permute outputs if requested:
n_nrecevier_mod_1282 = permuteOutputVar(n_nrecevier_mod_1282, outputDataPerms{1}, 4);
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
