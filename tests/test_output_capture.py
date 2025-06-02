import unittest
from unittest.mock import patch, MagicMock
import io
import contextlib
import logging
import sys
import os

# Imports for components to be tested or mocked
from llm_client import LLMCallStats, LLMClient
from llm_config import LLMConfig

from src.aot import orchestrator as aot_orchestrator_module
from aot.enums import AotTriggerMode, AssessmentDecision # Added AssessmentDecision for mocking
from aot.dataclasses import AoTRunnerConfig, Solution as AoTSolution
from aot.processor import AoTProcessor

from src.l2t import orchestrator as l2t_orchestrator_module
from src.l2t import processor as l2t_processor_module
from l2t.enums import L2TTriggerMode
from l2t.dataclasses import L2TConfig, L2TSolution, L2TResult, L2TModelConfigs

from src.hybrid import orchestrator as hybrid_orchestrator_module # Added
from src.hybrid import processor as hybrid_processor_module
from src.hybrid.enums import HybridTriggerMode # Changed to src.hybrid.enums
from src.hybrid.dataclasses import HybridConfig, HybridSolution, HybridResult # Changed to src.hybrid.dataclasses

from src.heuristic_detector import HeuristicDetector
from aot.dataclasses import ParsedLLMOutput


# Helper function to reconfigure logging for test methods that still use it
def reconfigure_logging_for_test(force=False):
    logging.shutdown()
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        handler.close()
    logging.basicConfig(level=logging.CRITICAL, stream=sys.stderr, format='%(levelname)s:%(name)s:%(message)s', force=force)

class TestOutputCapture(unittest.TestCase):

    def setUp(self):
        self.mock_llm_stats = MagicMock(spec=LLMCallStats, completion_tokens=10, prompt_tokens=10, call_duration_seconds=0.1, model_name='mock_model')
        self.mock_llm_config = MagicMock(spec=LLMConfig, temperature=0.0)

    def common_test_setup(self):
        mock_llm_client = MagicMock(spec=LLMClient)
        llm_stats = MagicMock(spec=LLMCallStats, completion_tokens=10, prompt_tokens=20,
                              call_duration_seconds=0.5, model_name="test_model_common")
        mock_llm_client.call.return_value = ("Mocked LLM Response", llm_stats)
        return mock_llm_client, llm_stats

    @patch.object(aot_orchestrator_module, 'AoTProcess', spec=True)
    def test_aot_orchestrator_no_default_std_out_err(self, MockAoTProcessClass):
        lib_logger = logging.getLogger('src')
        original_handlers = lib_logger.handlers[:]
        original_level = lib_logger.level
        original_propagate = lib_logger.propagate
        self.addCleanup(setattr, lib_logger, 'handlers', original_handlers)
        self.addCleanup(setattr, lib_logger, 'level', original_level)
        self.addCleanup(setattr, lib_logger, 'propagate', original_propagate)
        lib_logger.handlers = []
        lib_logger.addHandler(logging.NullHandler())
        lib_logger.setLevel(logging.DEBUG)
        lib_logger.propagate = False

        mock_llm_client = MagicMock(spec=LLMClient)
        mock_llm_client.call.return_value = ('Mocked LLM Response', self.mock_llm_stats)

        mock_aot_process_instance = MockAoTProcessClass.return_value
        mock_aot_process_instance.execute.return_value = None

        mock_solution = MagicMock(spec=AoTSolution)
        mock_solution.final_answer = 'Mock Answer'
        mock_solution.reasoning_trace = []
        aot_result_on_solution = MagicMock(succeeded=True, final_answer='Mock Answer')
        aot_result_on_solution.total_llm_interaction_time_seconds = 0.0
        aot_result_on_solution.total_completion_tokens = 0
        aot_result_on_solution.total_prompt_tokens = 0
        mock_solution.aot_result = aot_result_on_solution
        mock_solution.aot_summary_output = 'Mock AOT Summary'
        mock_solution.aot_failed_and_fell_back = False
        mock_solution.fallback_call_stats = None
        mock_solution.main_call_stats = None
        mock_solution.assessment_stats = None
        mock_solution.assessment_decision = None
        mock_solution.total_completion_tokens = 0
        mock_solution.total_prompt_tokens = 0
        mock_solution.grand_total_tokens = 0
        mock_solution.total_llm_interaction_time_seconds = 0.0
        mock_solution.total_wall_clock_time_seconds = 0.0

        mock_aot_process_instance.get_result.return_value = (mock_solution, 'Mock AoT Process Summary')

        mock_runner_config = MagicMock(spec=AoTRunnerConfig)
        mock_runner_config.main_model_names = ['mock_model']
        mock_runner_config.max_steps = 1
        mock_runner_config.max_reasoning_tokens = None
        mock_runner_config.max_time_seconds = 30
        mock_runner_config.no_progress_limit = 2
        mock_runner_config.pass_remaining_steps_pct = None

        orchestrator_AotTriggerMode = aot_orchestrator_module.AotTriggerMode

        orchestrator = aot_orchestrator_module.InteractiveAoTOrchestrator(
            llm_client=mock_llm_client,
            trigger_mode=orchestrator_AotTriggerMode.ALWAYS_AOT,
            aot_config=mock_runner_config,
            direct_oneshot_llm_config=self.mock_llm_config,
            assessment_llm_config=self.mock_llm_config,
            aot_main_llm_config=self.mock_llm_config,
            direct_oneshot_model_names=[],
            assessment_model_names=[],
            heuristic_detector=MagicMock(spec=HeuristicDetector)
        )

        self.assertIsNotNone(orchestrator.aot_process_instance, "AoTProcess instance should be initialized in ALWAYS_AOT mode")
        self.assertIs(orchestrator.aot_process_instance, mock_aot_process_instance, "orchestrator.aot_process_instance should be the mock_created_aot_process_instance")
        MockAoTProcessClass.assert_called_once()

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        try:
            with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
                orchestrator.solve('test problem')
        except Exception as e:
            self.fail(f"Test failed with unexpected exception: {e}\nSTDOUT:\n{stdout_capture.getvalue()}\nSTDERR:\n{stderr_capture.getvalue()}")

        self.assertEqual(stdout_capture.getvalue(), "", "STDOUT should be empty")
        self.assertEqual(stderr_capture.getvalue(), "", "STDERR should be empty")

        mock_aot_process_instance.execute.assert_called_once_with(
            problem_description='test problem',
            model_name='default_aot_model'
        )

    @patch.dict(os.environ, {"LOG_LEVEL": "CRITICAL"})
    def test_aot_processor_run_no_leak(self):
        reconfigure_logging_for_test(force=True)
        mock_llm_client, common_llm_stats = self.common_test_setup()
        mock_llm_client.call.return_value = ('DONE! Final Answer: Mock Answer', self.mock_llm_stats)

        mock_aot_runner_config = MagicMock(spec=AoTRunnerConfig)
        mock_aot_runner_config.main_model_names = ["test_model_for_processor"]
        mock_aot_runner_config.max_steps = 1
        mock_aot_runner_config.max_reasoning_tokens = None
        mock_aot_runner_config.max_time_seconds = 60
        mock_aot_runner_config.no_progress_limit = 2
        mock_aot_runner_config.pass_remaining_steps_pct = None

        processor = AoTProcessor(
            llm_client=mock_llm_client,
            runner_config=mock_aot_runner_config,
            llm_config=self.mock_llm_config
        )
        problem_text = 'test problem'
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
                processor.run(problem_text)
        except Exception as e:
            self.fail(f"Test failed with unexpected exception: {e}\nSTDOUT:\n{stdout_capture.getvalue()}\nSTDERR:\n{stderr_capture.getvalue()}")
        self.assertEqual(stdout_capture.getvalue(), "", "STDOUT should be empty")
        self.assertEqual(stderr_capture.getvalue(), "", "STDERR should be empty")

    @patch.object(l2t_orchestrator_module, 'LLMClient', spec=True)
    @patch.object(l2t_orchestrator_module, 'L2TProcess', spec=True)
    def test_l2t_orchestrator_no_default_std_out_err(self, MockL2TProcessClass, MockLLMClientClass):
        lib_logger = logging.getLogger('src')
        original_handlers = lib_logger.handlers[:]
        original_level = lib_logger.level
        original_propagate = lib_logger.propagate
        self.addCleanup(setattr, lib_logger, 'handlers', original_handlers)
        self.addCleanup(setattr, lib_logger, 'level', original_level)
        self.addCleanup(setattr, lib_logger, 'propagate', original_propagate)
        lib_logger.handlers = []
        lib_logger.addHandler(logging.NullHandler())
        lib_logger.setLevel(logging.DEBUG)
        lib_logger.propagate = False

        mock_llm_client_instance = MockLLMClientClass.return_value
        mock_llm_client_instance.call.return_value = ('mock response', self.mock_llm_stats)

        orchestrator_internal_L2TTriggerMode = l2t_orchestrator_module.L2TTriggerMode

        mock_created_l2t_process_instance = MockL2TProcessClass.return_value
        mock_created_l2t_process_instance.execute.return_value = None

        mock_l2t_solution = MagicMock(spec=L2TSolution)
        mock_l2t_solution.final_answer = 'Mock L2T Answer'
        mock_l2t_result_mock = MagicMock(spec=L2TResult, succeeded=True, error_message=None,
                                         total_llm_interaction_time_seconds=0.0, total_completion_tokens=0,
                                         total_prompt_tokens=0, total_llm_calls=0,
                                         total_process_wall_clock_time_seconds=0.0, reasoning_graph=None)
        mock_l2t_solution.l2t_result = mock_l2t_result_mock
        mock_l2t_solution.l2t_summary_output = 'L2T Summary'
        mock_l2t_solution.l2t_failed_and_fell_back = False
        mock_l2t_solution.fallback_call_stats = None
        mock_l2t_solution.main_call_stats = None
        mock_l2t_solution.assessment_stats = None
        mock_l2t_solution.assessment_decision = None
        mock_l2t_solution.total_completion_tokens = 0
        mock_l2t_solution.total_prompt_tokens = 0
        mock_l2t_solution.grand_total_tokens = 0
        mock_l2t_solution.total_llm_interaction_time_seconds = 0.0
        mock_l2t_solution.total_wall_clock_time_seconds = 0.0

        mock_created_l2t_process_instance.get_result.return_value = (mock_l2t_solution, 'Mocked L2T Process Summary')

        mock_l2t_config = MagicMock(spec=L2TConfig,
                                    initial_prompt_model_names=['mock'],
                                    classification_model_names=['mock'],
                                    thought_generation_model_names=['mock'],
                                    max_steps=1, max_total_nodes=1, max_time_seconds=10,
                                    x_fmt_default='', x_eva_default='', pass_remaining_steps_pct=None)

        mock_l2t_model_configs = MagicMock(spec=L2TModelConfigs,
                                           initial_thought_config=self.mock_llm_config,
                                           node_classification_config=self.mock_llm_config,
                                           node_thought_generation_config=self.mock_llm_config,
                                           orchestrator_oneshot_config=self.mock_llm_config,
                                           summary_config=self.mock_llm_config)

        orchestrator = l2t_orchestrator_module.L2TOrchestrator(
            trigger_mode=orchestrator_internal_L2TTriggerMode.ALWAYS_L2T,
            l2t_config=mock_l2t_config,
            model_configs=mock_l2t_model_configs,
            api_key="mock_api_key",
            heuristic_detector=MagicMock(spec=HeuristicDetector)
        )

        self.assertIsNotNone(orchestrator.l2t_process_instance)
        self.assertIs(orchestrator.l2t_process_instance, mock_created_l2t_process_instance)
        MockL2TProcessClass.assert_called_once()
        MockLLMClientClass.assert_called_once_with(api_key="mock_api_key", enable_rate_limiting=True, enable_audit_logging=True)

        initial_problem = 'test l2t problem'
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
                orchestrator.solve(initial_problem)
        except Exception as e:
            self.fail(f"Test failed with unexpected exception: {e}\nSTDOUT:\n{stdout_capture.getvalue()}\nSTDERR:\n{stderr_capture.getvalue()}")

        self.assertEqual(stdout_capture.getvalue(), "", "STDOUT should be empty")
        self.assertEqual(stderr_capture.getvalue(), "", "STDERR should be empty")
        mock_created_l2t_process_instance.execute.assert_called_once_with(
            problem_description=initial_problem, model_name='default_l2t_model'
        )

    @patch.object(l2t_processor_module, 'LLMClient', spec=True)
    @patch('src.l2t_processor_utils.node_processor.NodeProcessor', spec=True)
    def test_l2t_processor_run_no_leak(self, MockNodeProcessorClass, MockLLMClientClass):
        reconfigure_logging_for_test(force=True)

        mock_llm_client_instance = MockLLMClientClass.return_value
        initial_thought_stats = MagicMock(spec=LLMCallStats, completion_tokens=5, prompt_tokens=5, call_duration_seconds=0.1, model_name='initial_mock')
        mock_llm_client_instance.call.return_value = ('Initial thought content', initial_thought_stats)

        mock_node_processor_instance = MockNodeProcessorClass.return_value
        mock_node_processor_instance.process_node = MagicMock()
        mock_node_processor_instance._update_result_stats = MagicMock()

        mock_l2t_config = MagicMock(spec=L2TConfig)
        mock_l2t_config.initial_prompt_model_names = ["test_initial_model"]
        mock_l2t_config.classification_model_names = ["test_classification_model"]
        mock_l2t_config.thought_generation_model_names = ["test_thought_model"]
        mock_l2t_config.max_steps = 1
        mock_l2t_config.max_total_nodes = 10
        mock_l2t_config.max_time_seconds = 30
        mock_l2t_config.x_fmt_default = "fmt_default"
        mock_l2t_config.x_eva_default = "eva_default"
        mock_l2t_config.pass_remaining_steps_pct = None

        processor = l2t_processor_module.L2TProcessor(
            api_key='mock_key',
            l2t_config=mock_l2t_config,
            initial_thought_llm_config=self.mock_llm_config,
            node_processor_llm_config=self.mock_llm_config
        )

        problem_text = 'test l2t problem'
        with patch('src.l2t.processor.L2TResponseParser.parse_l2t_initial_response', return_value="Parsed initial thought") as mock_parser:
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()
            try:
                with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
                    result = processor.run(problem_text)
            except Exception as e:
                self.fail(f"Test failed with unexpected exception: {e}\nSTDOUT:\n{stdout_capture.getvalue()}\nSTDERR:\n{stderr_capture.getvalue()}")

        self.assertEqual(stdout_capture.getvalue(), "", "STDOUT should be empty")
        self.assertEqual(stderr_capture.getvalue(), "", "STDERR should be empty")
        mock_llm_client_instance.call.assert_called()
        mock_parser.assert_called_once_with('Initial thought content')
        if result and result.reasoning_graph and result.reasoning_graph.root_node_id and result.reasoning_graph.v_pres:
             mock_node_processor_instance.process_node.assert_called()
        else:
            logging.warning("L2TProcessor test: process_node not called, possibly due to empty graph after parsing.")

    @patch.object(hybrid_orchestrator_module, 'LLMClient', spec=True)
    @patch.object(hybrid_orchestrator_module, 'HybridProcess', spec=True)
    def test_hybrid_orchestrator_no_default_std_out_err(self, MockHybridProcessClass, MockLLMClientClass): # Corrected argument order
        lib_logger = logging.getLogger('src')
        original_handlers = lib_logger.handlers[:]
        original_level = lib_logger.level
        original_propagate = lib_logger.propagate
        self.addCleanup(setattr, lib_logger, 'handlers', original_handlers)
        self.addCleanup(setattr, lib_logger, 'level', original_level)
        self.addCleanup(setattr, lib_logger, 'propagate', original_propagate)
        lib_logger.handlers = []
        lib_logger.addHandler(logging.NullHandler())
        lib_logger.setLevel(logging.DEBUG)
        lib_logger.propagate = False

        mock_orchestrator_llm_client = MockLLMClientClass.return_value
        mock_orchestrator_llm_client.call.return_value = ('Mocked Orchestrator LLM Call', self.mock_llm_stats)

        # No longer need orchestrator_internal_HybridTriggerMode variable
        # Will use HybridTriggerMode directly as it's imported from src.hybrid.enums

        mock_created_hybrid_process_instance = MockHybridProcessClass.return_value
        mock_created_hybrid_process_instance.execute.return_value = None

        mock_hybrid_result = MagicMock(spec=HybridResult, succeeded=True, error_message=None,
                                       reasoning_call_stats=self.mock_llm_stats,
                                       response_call_stats=self.mock_llm_stats,
                                       extracted_reasoning='Mocked reasoning')
        mock_hybrid_solution = MagicMock(spec=HybridSolution, final_answer='Mock Hybrid Answer',
                                         hybrid_result=mock_hybrid_result,
                                         reasoning_trace=['Mocked reasoning'],
                                         hybrid_summary_output='Hybrid Summary',
                                         hybrid_failed_and_fell_back=False, fallback_call_stats=None,
                                         main_call_stats=None, assessment_stats=None, assessment_decision=None,
                                         total_completion_tokens=0, total_prompt_tokens=0, grand_total_tokens=0,
                                         total_llm_interaction_time_seconds=0.0, total_wall_clock_time_seconds=0.0)
        mock_created_hybrid_process_instance.get_result.return_value = (mock_hybrid_solution, 'Mocked Hybrid Process Summary')

        mock_hybrid_config = MagicMock(spec=HybridConfig,
                                       reasoning_model_name='mock_reasoning_model',
                                       reasoning_model_temperature=0.0,
                                       reasoning_prompt_template='{problem_description} {reasoning_complete_token}',
                                       reasoning_complete_token='[DONE]',
                                       response_model_name='mock_response_model',
                                       response_model_temperature=0.0,
                                       response_prompt_template='{problem_description} {extracted_reasoning}',
                                       max_reasoning_tokens=50, max_response_tokens=50)

        orchestrator = hybrid_orchestrator_module.HybridOrchestrator(
            trigger_mode=HybridTriggerMode.ALWAYS_HYBRID, # Use directly imported HybridTriggerMode
            hybrid_config=mock_hybrid_config,
            direct_oneshot_model_names=['mock_oneshot'],
            direct_oneshot_temperature=0.0,
            api_key='mock_api_key',
            assessment_model_names=None,
            assessment_temperature=None,
            heuristic_detector=MagicMock(spec=HeuristicDetector)
        )

        self.assertIsNotNone(orchestrator.hybrid_process_instance)
        self.assertIs(orchestrator.hybrid_process_instance, mock_created_hybrid_process_instance)
        MockHybridProcessClass.assert_called_once()
        MockLLMClientClass.assert_called_once_with(api_key='mock_api_key', enable_rate_limiting=True, enable_audit_logging=True)


        initial_problem = 'test hybrid problem'
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
                orchestrator.solve(initial_problem)
        except Exception as e:
            self.fail(f"Test failed with unexpected exception: {e}\nSTDOUT:\n{stdout_capture.getvalue()}\nSTDERR:\n{stderr_capture.getvalue()}")

        self.assertEqual(stdout_capture.getvalue(), "", "STDOUT should be empty")
        self.assertEqual(stderr_capture.getvalue(), "", "STDERR should be empty")
        mock_created_hybrid_process_instance.execute.assert_called_once_with(
            problem_description=initial_problem, model_name='default_hybrid_model'
        )

    @patch.object(hybrid_processor_module, 'LLMClient', spec=True)
    def test_hybrid_processor_run_no_leak(self, MockLLMClientClass):
        reconfigure_logging_for_test(force=True)
        mock_llm_client_instance = MockLLMClientClass.return_value
        mock_llm_client_instance.call.return_value = ('Mocked LLM Output', self.mock_llm_stats)

        mock_hybrid_config = MagicMock(spec=HybridConfig)
        mock_hybrid_config.reasoning_model_name = "test_reasoning_model"
        mock_hybrid_config.response_model_name = "test_response_model"
        mock_hybrid_config.reasoning_prompt_template = "Reason: {problem_description} {reasoning_complete_token}"
        mock_hybrid_config.reasoning_complete_token = "<DONE>"
        mock_hybrid_config.response_prompt_template = "Respond: {problem_description} {extracted_reasoning}"
        mock_hybrid_config.reasoning_model_temperature = 0.1
        mock_hybrid_config.max_reasoning_tokens = 100
        mock_hybrid_config.response_model_temperature = 0.7
        mock_hybrid_config.max_response_tokens = 100

        processor = hybrid_processor_module.HybridProcessor(
            llm_client=mock_llm_client_instance,
            config=mock_hybrid_config
        )

        problem_text = 'test hybrid problem'
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
                processor.run(problem_text)
        except Exception as e:
            self.fail(f"Test failed with unexpected exception: {e}\nSTDOUT:\n{stdout_capture.getvalue()}\nSTDERR:\n{stderr_capture.getvalue()}")

        self.assertEqual(stdout_capture.getvalue(), "", "STDOUT should be empty")
        self.assertEqual(stderr_capture.getvalue(), "", "STDERR should be empty")
        self.assertGreaterEqual(mock_llm_client_instance.call.call_count, 1)

if __name__ == '__main__':
    unittest.main()
