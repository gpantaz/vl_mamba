from collections.abc import Iterator
from random import random
from typing import Any, Optional, Union

import cv2
import numpy as np
from numpy import typing


class PlotBoundingBoxes:
    """Class for plotting bounding boxes on image.

    Adapted from https://github.com/pzzhang/VinVL.
    """

    def __init__(self) -> None:
        self._default_font = cv2.FONT_HERSHEY_SIMPLEX

    def draw_bb(
        self,
        image: typing.NDArray[np.uint8],
        boxes_coords: list[list[float]],
        boxes_labels: list[Any],
        probs: Optional[typing.NDArray[np.float32]] = None,
        draw_label: Optional[bool] = True,
    ) -> None:
        """Plot the bounding boxes."""
        self._width = image.shape[1]
        self._height = image.shape[0]
        self._font_info = self._get_font_info()

        color = self._get_color(boxes_labels)

        for idx, (rect, label) in enumerate(zip(boxes_coords, boxes_labels)):
            (start_point, end_point) = self._get_start_end_positions(rect)
            cv2.rectangle(
                image,
                start_point,
                end_point,
                color[label],
                self._font_info["font_thickness"],
            )
            if draw_label:
                rect_label = f"{label}"
                if probs is not None:
                    rect_label = f"{label}-{probs[idx]:.2f}"
                self._annotate(image, rect, rect_label, color[label])

    def _annotate(
        self,
        image: typing.NDArray[np.uint8],
        rect: list[float],
        rect_label: str,
        color_label: tuple[int, int, int],
    ) -> None:
        """Annotate a bounding box."""

        def gen_candidate() -> Iterator[tuple[int, int]]:  # noqa: WPS430
            """Get coordinates for text."""
            # above of top left
            yield int(rect[0]) + 2, int(rect[1]) - 4
            # below of bottom left
            yield int(rect[0]) + 2, int(rect[3]) + text_height + 2  # noqa: WPS221

        (_, text_height), _ = cv2.getTextSize(
            rect_label,
            self._font_info["font"],
            self._font_info["font_scale"],
            self._font_info["font_thickness"],
        )
        for text_left, text_bottom in gen_candidate():
            if (  # noqa: WPS337
                0 <= text_left < self._width - 12  # noqa: WPS432
                and 12 < text_bottom < self._height  # noqa: WPS432
            ):
                self._put_text(
                    image=image,
                    text=rect_label,
                    bottomleft=(text_left, text_bottom),
                    color=color_label,
                )
                break

    def _get_start_end_positions(
        self, rect: list[float]
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """Get start and end positions for a bounding box."""
        start = (int(rect[0]), int(rect[1]))
        end = (int(rect[2]), int(rect[3]))
        return (start, end)

    def _get_color(self, boxes_labels: list[Any]) -> dict[Any, tuple[int, int, int]]:
        """Get label color for a bounding box."""
        color: dict[Any, tuple[int, int, int]] = {}
        dist_label = set(boxes_labels)
        for label in dist_label:
            color_candidate = (
                int(random() * 255),  # noqa: WPS432
                int(random() * 255),  # noqa: WPS432
                int(random() * 255),  # noqa: WPS432
            )
            while True:
                if color_candidate not in color.keys():
                    color[label] = color_candidate
                    break
        return color

    def _get_font_info(self) -> dict[str, Union[int, float]]:
        """Get font."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        ref = (self._height + self._width) / 2
        font_scale = ref / 500
        font_thickness = int(max(ref / 400, 1))  # noqa: WPS432
        return {
            "font": font,
            "font_scale": font_scale,
            "font_thickness": font_thickness,
        }

    def _put_text(
        self,
        image: typing.NDArray[np.uint8],
        text: str,
        bottomleft: tuple[int, int] = (0, 100),
        color: tuple[int, int, int] = (255, 255, 255),
    ) -> None:
        """Write text to image."""
        cv2.putText(
            image,
            text,
            bottomleft,
            self._default_font,
            self._font_info["font_scale"],
            color,
            self._font_info["font_thickness"],
        )
