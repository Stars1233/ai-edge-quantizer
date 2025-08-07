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

"""Test that the quantizer works for the model with -bloat16 inf min."""

import os

from absl.testing import parameterized

from tensorflow.python.platform import googletest
from ai_edge_quantizer import qtyping
from ai_edge_quantizer import quantizer
from ai_edge_quantizer.utils import test_utils


_TEST_MODEL_FOLDER = test_utils.get_path_to_datafile('models/')
_QuantAlgo = quantizer.AlgorithmName


class PadV2InfMaxPool2DTest(test_utils.BaseOpTestCase):

  def setUp(self):
    super().setUp()
    self._op_name = qtyping.TFLOperationName.MAX_POOL_2D

  @parameterized.parameters(
      # algorithm_key, activations_num_bits, symmetric
      (_QuantAlgo.MIN_MAX_UNIFORM_QUANT, 8, True),
      (_QuantAlgo.MIN_MAX_UNIFORM_QUANT, 8, False),
      (_QuantAlgo.MIN_MAX_UNIFORM_QUANT, 16, True),
      (_QuantAlgo.OCTAV, 8, True),
      (_QuantAlgo.OCTAV, 16, True),
  )
  def test_static_quantization_accuracy_and_size_within_tolerance(
      self, algorithm_key, activations_num_bits, symmetric
  ):
    output_tolerance = 1e-2
    model_filename = 'padv2_inf_max_pool_2d.tflite'
    model_path = os.path.join(_TEST_MODEL_FOLDER, model_filename)

    activation_config = test_utils.get_static_activation_quant_setting(
        activations_num_bits, symmetric
    )
    op_config = test_utils.get_static_op_quant_config(activation_config)
    self.assert_quantization_accuracy(
        algorithm_key=algorithm_key,
        model_path=model_path,
        op_name=self._op_name,
        op_config=op_config,
        output_tolerance=output_tolerance,
    )


if __name__ == '__main__':
  googletest.main()
