# Copyright 2024 The AI Edge Quantizer Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os

from absl.testing import parameterized
import numpy as np

from tensorflow.python.platform import googletest
from ai_edge_quantizer import qtyping
from ai_edge_quantizer.algorithms.uniform_quantize import common_quantize
from ai_edge_quantizer.algorithms.uniform_quantize.naive_min_max_quantize_op_tests import test_utils as naive_min_max_test_utils
from ai_edge_quantizer.utils import test_utils
from ai_edge_quantizer.utils import tfl_flatbuffer_utils

_TFLOpName = qtyping.TFLOperationName
_ComputePrecision = qtyping.ComputePrecision
_TensorQuantConfig = qtyping.TensorQuantizationConfig
_QuantTransformation = qtyping.QuantTransformation
_OpTestInfo = naive_min_max_test_utils.OpTestInfo

_TEST_DATA_PREFIX_PATH = test_utils.get_path_to_datafile(
    "../../../tests/models"
)
_DEFAULT_ACTIVATION_QUANT_SETTING = (
    naive_min_max_test_utils.DEFAULT_ACTIVATION_QUANT_SETTING
)
_DEFAULT_WEIGHT_QUANT_SETTING = (
    naive_min_max_test_utils.DEFAULT_WEIGHT_QUANT_SETTING
)


class SplitTest(naive_min_max_test_utils.NaiveMinMaxQuantizeTest):

  def setUp(self):
    super().setUp()
    np.random.seed(666)
    self._test_model_path = os.path.join(
        _TEST_DATA_PREFIX_PATH, "single_split.tflite"
    )
    self._op_test_info = _OpTestInfo(
        test_model=tfl_flatbuffer_utils.read_model(self._test_model_path),
        op_tensor_names={},
        input_range=(np.array([[-10]]), np.array([[8]])),
        output_range=(np.array([[10]]), np.array([[88]])),
    )
    # The test model has one subgraph for now.
    self._graph_info = qtyping.GraphInfo(
        subgraph_tensors=self._op_test_info.test_model.subgraphs[0].tensors,
        buffers=self._op_test_info.test_model.buffers,
    )

  @parameterized.parameters(
      (_DEFAULT_ACTIVATION_QUANT_SETTING),
      (
          _TensorQuantConfig(
              num_bits=16,
              symmetric=True,
              granularity=qtyping.QuantGranularity.TENSORWISE,
          )
      ),
  )
  def test_materialize_split_succeeds(self, activation_tensor_config):
    op_quant_config = qtyping.OpQuantizationConfig(
        activation_tensor_config=activation_tensor_config,
        weight_tensor_config=_DEFAULT_WEIGHT_QUANT_SETTING,
        compute_precision=_ComputePrecision.INTEGER,  # SRQ.
    )
    # Read from Model Explorer.
    subgraph0 = self._op_test_info.test_model.subgraphs[0]
    subgraph_op_id = 0
    op = subgraph0.operators[subgraph_op_id]
    op_info = qtyping.OpInfo(
        op=op,
        op_name=qtyping.TFLOperationName.SPLIT,
        subgraph_op_index=subgraph_op_id,
        op_quant_config=op_quant_config,
    )

    # Test settings.
    op_tensor_names = {}
    op_tensor_names["input"] = "model/tf.split/split/split_dim"
    op_tensor_names["input2"] = "serving_default_input_1:0"
    op_tensor_names["output"] = "PartitionedCall:0"
    op_tensor_names["output2"] = "PartitionedCall:1"
    self._op_test_info.op_tensor_names = op_tensor_names
    self._test_no_weights_op(
        op_info,
        self._graph_info,
        self._op_test_info,
        common_quantize.materialize_split,
        same_input_output_params=True,
        inputs_to_ignore=[0],  # Ignore split dimension tensor.
    )


if __name__ == "__main__":
  googletest.main()
