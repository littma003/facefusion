from argparse import ArgumentParser
from functools import lru_cache
from typing import Any, List, Tuple

import cv2
import numpy as np

import facefusion.choices
import facefusion.jobs.job_manager
import facefusion.jobs.job_store
import facefusion.processors.core as processors
from facefusion import config, content_analyser, face_classifier, face_detector, face_landmarker, face_masker, face_recognizer, inference_manager, logger, process_manager, state_manager, video_manager, wording
from facefusion.common_helper import get_first
from facefusion.download import conditional_download_hashes, conditional_download_sources, resolve_download_url
from facefusion.execution import has_execution_provider
from facefusion.face_analyser import get_average_face, get_many_faces, get_one_face
from facefusion.face_helper import paste_back, warp_face_by_face_landmark_5
from facefusion.face_masker import create_area_mask, create_box_mask, create_occlusion_mask, create_region_mask
from facefusion.face_selector import find_similar_faces, sort_and_filter_faces, sort_faces_by_order
from facefusion.face_store import get_reference_faces
from facefusion.filesystem import filter_image_paths, has_image, in_directory, is_image, is_video, resolve_relative_path, same_file_extension
from facefusion.model_helper import get_static_model_initializer
from facefusion.processors import choices as processors_choices
from facefusion.processors.pixel_boost import explode_pixel_boost, implode_pixel_boost
from facefusion.processors.types import FaceSwapperInputs
from facefusion.program_helper import find_argument_group
from facefusion.thread_helper import conditional_thread_semaphore
from facefusion.types import ApplyStateItem, Args, DownloadScope, Embedding, Face, InferencePool, ModelOptions, ModelSet, ProcessMode, QueuePayload, UpdateProgress, VisionFrame
from facefusion.vision import read_image, read_static_image, read_static_images, unpack_resolution, write_image


@lru_cache(maxsize = None)
def create_static_model_set(download_scope : DownloadScope) -> ModelSet:
	return\
	{
		'blendswap_256':
		{
			'hashes':
			{
				'face_swapper':
				{
					'url': resolve_download_url('models-3.0.0', 'blendswap_256.hash'),
					'path': resolve_relative_path('../.assets/models/blendswap_256.hash')
				}
			},
			'sources':
			{
				'face_swapper':
				{
					'url': resolve_download_url('models-3.0.0', 'blendswap_256.onnx'),
					'path': resolve_relative_path('../.assets/models/blendswap_256.onnx')
				}
			},
			'type': 'blendswap',
			'template': 'ffhq_512',
			'size': (256, 256),
			'mean': [ 0.0, 0.0, 0.0 ],
			'standard_deviation': [ 1.0, 1.0, 1.0 ]
		},
		'ghost_1_256':
		{
			'hashes':
			{
				'face_swapper':
				{
					'url': resolve_download_url('models-3.0.0', 'ghost_1_256.hash'),
					'path': resolve_relative_path('../.assets/models/ghost_1_256.hash')
				},
				'embedding_converter':
				{
					'url': resolve_download_url('models-3.0.0', 'arcface_converter_ghost.hash'),
					'path': resolve_relative_path('../.assets/models/arcface_converter_ghost.hash')
				}
			},
			'sources':
			{
				'face_swapper':
				{
					'url': resolve_download_url('models-3.0.0', 'ghost_1_256.onnx'),
					'path': resolve_relative_path('../.assets/models/ghost_1_256.onnx')
				},
				'embedding_converter':
				{
					'url': resolve_download_url('models-3.0.0', 'arcface_converter_ghost.onnx'),
					'path': resolve_relative_path('../.assets/models/arcface_converter_ghost.onnx')
				}
			},
			'type': 'ghost',
			'template': 'arcface_112_v1',
			'size': (256, 256),
			'mean': [ 0.5, 0.5, 0.5 ],
			'standard_deviation': [ 0.5, 0.5, 0.5 ]
		},
		'ghost_2_256':
		{
			'hashes':
			{
				'face_swapper':
				{
					'url': resolve_download_url('models-3.0.0', 'ghost_2_256.hash'),
					'path': resolve_relative_path('../.assets/models/ghost_2_256.hash')
				},
				'embedding_converter':
				{
					'url': resolve_download_url('models-3.0.0', 'arcface_converter_ghost.hash'),
					'path': resolve_relative_path('../.assets/models/arcface_converter_ghost.hash')
				}
			},
			'sources':
			{
				'face_swapper':
				{
					'url': resolve_download_url('models-3.0.0', 'ghost_2_256.onnx'),
					'path': resolve_relative_path('../.assets/models/ghost_2_256.onnx')
				},
				'embedding_converter':
				{
					'url': resolve_download_url('models-3.0.0', 'arcface_converter_ghost.onnx'),
					'path': resolve_relative_path('../.assets/models/arcface_converter_ghost.onnx')
				}
			},
			'type': 'ghost',
			'template': 'arcface_112_v1',
			'size': (256, 256),
			'mean': [ 0.5, 0.5, 0.5 ],
			'standard_deviation': [ 0.5, 0.5, 0.5 ]
		},
		'ghost_3_256':
		{
			'hashes':
			{
				'face_swapper':
				{
					'url': resolve_download_url('models-3.0.0', 'ghost_3_256.hash'),
					'path': resolve_relative_path('../.assets/models/ghost_3_256.hash')
				},
				'embedding_converter':
				{
					'url': resolve_download_url('models-3.0.0', 'arcface_converter_ghost.hash'),
					'path': resolve_relative_path('../.assets/models/arcface_converter_ghost.hash')
				}
			},
			'sources':
			{
				'face_swapper':
				{
					'url': resolve_download_url('models-3.0.0', 'ghost_3_256.onnx'),
					'path': resolve_relative_path('../.assets/models/ghost_3_256.onnx')
				},
				'embedding_converter':
				{
					'url': resolve_download_url('models-3.0.0', 'arcface_converter_ghost.onnx'),
					'path': resolve_relative_path('../.assets/models/arcface_converter_ghost.onnx')
				}
			},
			'type': 'ghost',
			'template': 'arcface_112_v1',
			'size': (256, 256),
			'mean': [ 0.5, 0.5, 0.5 ],
			'standard_deviation': [ 0.5, 0.5, 0.5 ]
		},
		'hififace_unofficial_256':
		{
			'hashes':
			{
				'face_swapper':
				{
					'url': resolve_download_url('models-3.1.0', 'hififace_unofficial_256.hash'),
					'path': resolve_relative_path('../.assets/models/hififace_unofficial_256.hash')
				},
				'embedding_converter':
				{
					'url': resolve_download_url('models-3.1.0', 'arcface_converter_hififace.hash'),
					'path': resolve_relative_path('../.assets/models/arcface_converter_hififace.hash')
				}
			},
			'sources':
			{
				'face_swapper':
				{
					'url': resolve_download_url('models-3.1.0', 'hififace_unofficial_256.onnx'),
					'path': resolve_relative_path('../.assets/models/hififace_unofficial_256.onnx')
				},
				'embedding_converter':
				{
					'url': resolve_download_url('models-3.1.0', 'arcface_converter_hififace.onnx'),
					'path': resolve_relative_path('../.assets/models/arcface_converter_hififace.onnx')
				}
			},
			'type': 'hififace',
			'template': 'mtcnn_512',
			'size': (256, 256),
			'mean': [ 0.5, 0.5, 0.5 ],
			'standard_deviation': [ 0.5, 0.5, 0.5 ]
		},
		'hyperswap_1a_256':
		{
			'hashes':
			{
				'face_swapper':
				{
					'url': resolve_download_url('models-3.3.0', 'hyperswap_1a_256.hash'),
					'path': resolve_relative_path('../.assets/models/hyperswap_1a_256.hash')
				}
			},
			'sources':
			{
				'face_swapper':
				{
					'url': resolve_download_url('models-3.3.0', 'hyperswap_1a_256.onnx'),
					'path': resolve_relative_path('../.assets/models/hyperswap_1a_256.onnx')
				}
			},
			'type': 'hyperswap',
			'template': 'arcface_128',
			'size': (256, 256),
			'mean': [ 0.5, 0.5, 0.5 ],
			'standard_deviation': [ 0.5, 0.5, 0.5 ]
		},
		'hyperswap_1b_256':
		{
			'hashes':
			{
				'face_swapper':
				{
					'url': resolve_download_url('models-3.3.0', 'hyperswap_1b_256.hash'),
					'path': resolve_relative_path('../.assets/models/hyperswap_1b_256.hash')
				}
			},
			'sources':
				{
					'face_swapper':
					{
						'url': resolve_download_url('models-3.3.0', 'hyperswap_1b_256.onnx'),
						'path': resolve_relative_path('../.assets/models/hyperswap_1b_256.onnx')
					}
				},
			'type': 'hyperswap',
			'template': 'arcface_128',
			'size': (256, 256),
			'mean': [ 0.5, 0.5, 0.5 ],
			'standard_deviation': [ 0.5, 0.5, 0.5 ]
		},
		'hyperswap_1c_256':
		{
			'hashes':
			{
				'face_swapper':
				{
					'url': resolve_download_url('models-3.3.0', 'hyperswap_1c_256.hash'),
					'path': resolve_relative_path('../.assets/models/hyperswap_1c_256.hash')
				}
			},
			'sources':
			{
				'face_swapper':
				{
					'url': resolve_download_url('models-3.3.0', 'hyperswap_1c_256.onnx'),
					'path': resolve_relative_path('../.assets/models/hyperswap_1c_256.onnx')
				}
			},
			'type': 'hyperswap',
			'template': 'arcface_128',
			'size': (256, 256),
			'mean': [ 0.5, 0.5, 0.5 ],
			'standard_deviation': [ 0.5, 0.5, 0.5 ]
		},
		'inswapper_128':
		{
			'hashes':
			{
				'face_swapper':
				{
					'url': resolve_download_url('models-3.0.0', 'inswapper_128.hash'),
					'path': resolve_relative_path('../.assets/models/inswapper_128.hash')
				}
			},
			'sources':
			{
				'face_swapper':
				{
					'url': resolve_download_url('models-3.0.0', 'inswapper_128.onnx'),
					'path': resolve_relative_path('../.assets/models/inswapper_128.onnx')
				}
			},
			'type': 'inswapper',
			'template': 'arcface_128',
			'size': (128, 128),
			'mean': [ 0.0, 0.0, 0.0 ],
			'standard_deviation': [ 1.0, 1.0, 1.0 ]
		},
		'inswapper_128_fp16':
		{
			'hashes':
			{
				'face_swapper':
				{
					'url': resolve_download_url('models-3.0.0', 'inswapper_128_fp16.hash'),
					'path': resolve_relative_path('../.assets/models/inswapper_128_fp16.hash')
				}
			},
			'sources':
			{
				'face_swapper':
				{
					'url': resolve_download_url('models-3.0.0', 'inswapper_128_fp16.onnx'),
					'path': resolve_relative_path('../.assets/models/inswapper_128_fp16.onnx')
				}
			},
			'type': 'inswapper',
			'template': 'arcface_128',
			'size': (128, 128),
			'mean': [ 0.0, 0.0, 0.0 ],
			'standard_deviation': [ 1.0, 1.0, 1.0 ]
		},
		'simswap_256':
		{
			'hashes':
			{
				'face_swapper':
				{
					'url': resolve_download_url('models-3.0.0', 'simswap_256.hash'),
					'path': resolve_relative_path('../.assets/models/simswap_256.hash')
				},
				'embedding_converter':
				{
					'url': resolve_download_url('models-3.0.0', 'arcface_converter_simswap.hash'),
					'path': resolve_relative_path('../.assets/models/arcface_converter_simswap.hash')
				}
			},
			'sources':
			{
				'face_swapper':
				{
					'url': resolve_download_url('models-3.0.0', 'simswap_256.onnx'),
					'path': resolve_relative_path('../.assets/models/simswap_256.onnx')
				},
				'embedding_converter':
				{
					'url': resolve_download_url('models-3.0.0', 'arcface_converter_simswap.onnx'),
					'path': resolve_relative_path('../.assets/models/arcface_converter_simswap.onnx')
				}
			},
			'type': 'simswap',
			'template': 'arcface_112_v1',
			'size': (256, 256),
			'mean': [ 0.485, 0.456, 0.406 ],
			'standard_deviation': [ 0.229, 0.224, 0.225 ]
		},
		'simswap_unofficial_512':
		{
			'hashes':
			{
				'face_swapper':
				{
					'url': resolve_download_url('models-3.0.0', 'simswap_unofficial_512.hash'),
					'path': resolve_relative_path('../.assets/models/simswap_unofficial_512.hash')
				},
				'embedding_converter':
				{
					'url': resolve_download_url('models-3.0.0', 'arcface_converter_simswap.hash'),
					'path': resolve_relative_path('../.assets/models/arcface_converter_simswap.hash')
				}
			},
			'sources':
			{
				'face_swapper':
				{
					'url': resolve_download_url('models-3.0.0', 'simswap_unofficial_512.onnx'),
					'path': resolve_relative_path('../.assets/models/simswap_unofficial_512.onnx')
				},
				'embedding_converter':
				{
					'url': resolve_download_url('models-3.0.0', 'arcface_converter_simswap.onnx'),
					'path': resolve_relative_path('../.assets/models/arcface_converter_simswap.onnx')
				}
			},
			'type': 'simswap',
			'template': 'arcface_112_v1',
			'size': (512, 512),
			'mean': [ 0.0, 0.0, 0.0 ],
			'standard_deviation': [ 1.0, 1.0, 1.0 ]
		},
		'uniface_256':
		{
			'hashes':
			{
				'face_swapper':
				{
					'url': resolve_download_url('models-3.0.0', 'uniface_256.hash'),
					'path': resolve_relative_path('../.assets/models/uniface_256.hash')
				}
			},
			'sources':
			{
				'face_swapper':
				{
					'url': resolve_download_url('models-3.0.0', 'uniface_256.onnx'),
					'path': resolve_relative_path('../.assets/models/uniface_256.onnx')
				}
			},
			'type': 'uniface',
			'template': 'ffhq_512',
			'size': (256, 256),
			'mean': [ 0.5, 0.5, 0.5 ],
			'standard_deviation': [ 0.5, 0.5, 0.5 ]
		}
	}


# 推理池管理
def get_inference_pool() -> Any:
    return None

def clear_inference_pool() -> None:
    return

# 参数处理
def register_args(parser : Any) -> None:
    return

def apply_args(args: Any, apply_state_item: Any) -> None:
    return

# 预处理与检查
def pre_check() -> bool:
    return True

def pre_process(input_image : np.ndarray) -> np.ndarray:
    return input_image

def post_process(input_image : np.ndarray) -> np.ndarray:
    return input_image

# 参考帧处理
def get_reference_frame() -> np.ndarray:
    return np.zeros((640, 480, 3), dtype=np.uint8)

# 核心处理方法
def process_frame(input_image : np.ndarray, face : Any, reference_face : Any) -> np.ndarray:
    return input_image

def process_frames(source_paths : List[str], queue_payloads : List[Any], update_progress : Any) -> None:
    return

def process_image(input_image : np.ndarray) -> np.ndarray:
    return input_image

def process_video(source_paths : List[str], temp_frame_paths : List[str]) -> None:
    return
