"""
main.py

Пример лабораторной работы по курсу "Технологии программирования на Python".

Модуль предназначен для демонстрации работы с обработкой изображений с помощью библиотеки OpenCV.
Реализован консольный интерфейс для применения различных методов обработки к изображению:
- обнаружение границ (edges)
- обнаружение углов (corners)
- обнаружение окружностей (circles)

Запуск:
    python main.py <метод> <путь_к_изображению> [-o путь_для_сохранения]

Аргументы:
    метод: edges | corners | circles
    путь_к_изображению: путь к входному изображению
    -o, --output: путь для сохранения результата (по умолчанию: <имя_входного_файла>_result.png)

Пример:
    python main.py edges input.jpg
    python main.py corners input.jpg -o corners_result.png

Автор: [Ваше имя]
"""

import argparse
import os
import numpy as np

import cv2

from implementation import ImageProcessing


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Обработка изображения с помощью методов ImageProcessing (OpenCV).",
    )
    parser.add_argument(
        "method",
        choices=[
            "edges",
            "corners",
            "circles",
            "conv",
            "gray",
            "gamma"
        ],
        help="Метод обработки: edges, corners, circles, conv, gray, gamma",
    )
    parser.add_argument(
        "input",
        help="Путь к входному изображению",
    )
    parser.add_argument(
        "-o", "--output",
        help="Путь для сохранения результата (по умолчанию: <input>_result.png)",
    )

    args = parser.parse_args()

    # Загрузка изображения
    image = cv2.imread(args.input)
    if image is None:
        print(f"Ошибка: не удалось загрузить изображение {args.input}")
        return

    processor = ImageProcessing()

    """
    Примеры ядер свёртки:
    np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]), # Собель по X
    np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]), # Собель по Y
    np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]), # Лаплас
    np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9,# Усредняющий
    np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]), # Резкость
    np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16, # Гаусс
    """

    # Выбор метода
    if args.method == "edges":
        result = processor.edge_detection(image)
    elif args.method == "corners":
        result = processor.corner_detection(image)
    elif args.method == "circles":
        result = processor.circle_detection(image)
    elif args.method == "conv":
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])  # Ядро для повышения резкости
        result = processor.convolution(image, kernel)
    elif args.method == "gray":
        result = processor.rgb_to_grayscale(image)
    elif args.method == "gamma":
        gamma_value = 2.2  # Пример значения гаммы
        result = processor.gamma_correction(image, gamma_value)
    else:
        print("Ошибка: неизвестный метод")
        return

    # Определение пути для сохранения
    if args.output:
        output_path = args.output
    else:
        base, ext = os.path.splitext(args.input)
        output_path = f"{base}_result.png"

    # Сохранение результата
    cv2.imwrite(output_path, result)
    print(f"Результат сохранён в {output_path}")


if __name__ == "__main__":
    main()
