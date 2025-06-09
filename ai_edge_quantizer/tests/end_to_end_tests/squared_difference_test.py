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

"""E2E tests for the quantizer for model with squared difference."""

import os

from absl.testing import parameterized

from tensorflow.python.platform import googletest
from ai_edge_quantizer import qtyping
from ai_edge_quantizer import quantizer
from ai_edge_quantizer.utils import test_utils


_TEST_MODEL_FOLDER = test_utils.get_path_to_datafile('../models/')


class SquaredDifferenceTest(test_utils.BaseOpTestCase):

  def setUp(self):
    super().setUp()
    self._op_name = qtyping.TFLOperationName.SQUARED_DIFFERENCE

  @parameterized.product(
      algorithm_key=[
          quantizer.AlgorithmName.MIN_MAX_UNIFORM_QUANT,
          quantizer.AlgorithmName.OCTAV,
      ],
      activations_num_bits_and_symmetric=[
          (8, True),
          (8, False),
      ],
  )
  def test_static_quantization_accuracy_and_size_within_tolerance(
      self, algorithm_key, activations_num_bits_and_symmetric
  ):
    output_tolerance = 1e-3
    model_filename = 'single_squared_difference.tflite'
    model_path = os.path.join(_TEST_MODEL_FOLDER, model_filename)

    activation_config = test_utils.get_static_activation_quant_setting(
        *activations_num_bits_and_symmetric
    )
    op_config = test_utils.get_static_op_quant_config(activation_config)
    # Not checking the model size reduction because this single op model is
    # expected to have larger size after static quantization.
    # Had to set num_validation_samples and num_calibration_samples to 1 because
    # the moving_average_update function with random calibration data produces
    # QSV's leading to a large error.
    self.assert_quantization_accuracy(
        algorithm_key=algorithm_key,
        model_path=model_path,
        op_name=self._op_name,
        op_config=op_config,
        num_validation_samples=1,
        num_calibration_samples=1,
        output_tolerance=output_tolerance,
    )


if __name__ == '__main__':
  googletest.main()
