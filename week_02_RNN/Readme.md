## Материалы модуля 2

<div align="center">
  <img src="../images/dls.png">
</div>

###  Рекуррентные нейронные сети (RNN).

В этом модуле вы узнаете, что такое рекуррентные нейронные сети, а также научитесь применять их для решения задачи классификации текстов.


### Лекция

<div style="display: flex; gap: 8px; align-items: baseline; white-space: nowrap; line-height: 1.2; margin-bottom: 16px;">
  <span>Запись лекции &laquo;Лекция. Рекуррентная нейронная сеть&raquo; доступна на</span>
  <a href="https://youtu.be/3OljkWQ2Uc0?si=zI2kFuqgBd0xIyxI" target="_blank" rel="noopener" aria-label="Watch on YouTube" style="text-decoration: none;">
    <img src="https://cdn.simpleicons.org/youtube" alt="YouTube" width="20" style="vertical-align: bottom;"> YouTube
  </a>
  <span>и</span>
  <a href="https://vkvideo.ru/video-155161349_456239185" target="_blank" rel="noopener" aria-label="Watch on VK" style="text-decoration: none;">
    <img src="https://cdn.simpleicons.org/vk" alt="VK" width="20" style="vertical-align: bottom;"> VK Видео
  </a>
</div>

<div style="display: flex; gap: 8px; align-items: baseline; white-space: nowrap; line-height: 1.2; margin-bottom: 16px;">
  <span>Запись лекции &laquo;Лекция. Обучение рекуррентной нейронной сети&raquo; доступна на</span>
  <a href="https://youtu.be/WsXgPD4lmlc?si=cu6cN6YbaBriB-hZ" target="_blank" rel="noopener" aria-label="Watch on YouTube" style="text-decoration: none;">
    <img src="https://cdn.simpleicons.org/youtube" alt="YouTube" width="20" style="vertical-align: bottom;"> YouTube
  </a>
  <span>и</span>
  <a href="https://vkvideo.ru/video-155161349_456239186" target="_blank" rel="noopener" aria-label="Watch on VK" style="text-decoration: none;">
    <img src="https://cdn.simpleicons.org/vk" alt="VK" width="20" style="vertical-align: bottom;"> VK Видео
  </a>
</div>

<div style="display: flex; gap: 8px; align-items: baseline; white-space: nowrap; line-height: 1.2; margin-bottom: 16px;">
  <span>Запись лекции &laquo;Лекция. Функции активации в RNN&raquo; доступна на</span>
  <a href="https://youtu.be/OF_JnQlHXQ4?si=Ja7LyISQvjjc27TR" target="_blank" rel="noopener" aria-label="Watch on YouTube" style="text-decoration: none;">
    <img src="https://cdn.simpleicons.org/youtube" alt="YouTube" width="20" style="vertical-align: bottom;"> YouTube
  </a>
  <span>и</span>
  <a href="https://vkvideo.ru/video-155161349_456239187" target="_blank" rel="noopener" aria-label="Watch on VK" style="text-decoration: none;">
    <img src="https://cdn.simpleicons.org/vk" alt="VK" width="20" style="vertical-align: bottom;"> VK Видео
  </a>
</div>

<div style="display: flex; gap: 8px; align-items: baseline; white-space: nowrap; line-height: 1.2; margin-bottom: 16px;">
  <span>Запись лекции &laquo;Лекция. LSTM, GRU&raquo; доступна на</span>
  <a href="https://youtu.be/2a_7SmTNrJA?si=7Teqw2SeOUJPqb-j" target="_blank" rel="noopener" aria-label="Watch on YouTube" style="text-decoration: none;">
    <img src="https://cdn.simpleicons.org/youtube" alt="YouTube" width="20" style="vertical-align: bottom;"> YouTube
  </a>
  <span>и</span>
  <a href="https://vkvideo.ru/video-155161349_456239188" target="_blank" rel="noopener" aria-label="Watch on VK" style="text-decoration: none;">
    <img src="https://cdn.simpleicons.org/vk" alt="VK" width="20" style="vertical-align: bottom;"> VK Видео
  </a>
</div>

В этом модуле мы поговорим о рекурентных нейронных сетях (RNN), а также об их улучшениях, таких как Bidirectional RNN, GRU и LSTM

Занятие ведёт Татьяна Гайнцева.


### Семинар
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DeepLearningSchool/part_2_nlp/blob/main/week_02_RNN/Practice/neural_text_classification.ipynb)

<div style="display: flex; gap: 8px; align-items: baseline; white-space: nowrap; line-height: 1.2; margin-bottom: 16px;">
  <span>Запись Семинара &laquo;Семинар. Решение задачи классификации текста&raquo; доступна на</span>
  <a href="https://youtu.be/ltyWeIPrAVA?si=WqMFMa2jMY43w055" target="_blank" rel="noopener" aria-label="Watch on YouTube" style="text-decoration: none;">
    <img src="https://cdn.simpleicons.org/youtube" alt="YouTube" width="20" style="vertical-align: bottom;"> YouTube
  </a>
  <span>и</span>
  <a href="https://vkvideo.ru/video-155161349_456239184" target="_blank" rel="noopener" aria-label="Watch on VK" style="text-decoration: none;">
    <img src="https://cdn.simpleicons.org/vk" alt="VK" width="20" style="vertical-align: bottom;"> VK Видео
  </a>
</div>


В этом видео мы решим задачу бинарной классификации с помощью рекуррентной нейронной сети. Узнаем, как использовать стандартные модули из библиотеки torch  и сравним два подхода к агрегации эмбеддингов: max pooling и простое усреднение.

Занятие ведет Антон Земеров.

### Домашнее задание

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DeepLearningSchool/part_2_nlp/blob/main/week_02_RNN/Homework/hw_2_text_classification.ipynb)

В этом домашнем задании вам предстоит решить задачу классификации текстов новостей по темам.

