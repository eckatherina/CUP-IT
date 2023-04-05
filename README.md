# CUP-IT
Case championship вместе с @akscent

## Задачи кейса
1. Обучить модель ранжировать текстовые комментарии в порядке их популярности.
2. Сформулировать инсайты о том, что обычно содержит популярный комментарий (in english), чтобы команда VK могла использовать эту информацию для улучшения комментариев своих пользователей. 
3. Предложить методы взаимодействия с комментаторами: механизмы общения с популярными и непопулярными комментариями.

## EDA 
Дан текст на английском языке - это упрощает задачу, поскольку существует множество пред-обученных моделей, подходящих для задач ранжирования: семейства моделей BERT, BART, GPT, USE, DeepPavlov.
Все они основаны на векторизации/токенизации и нейронных сетях. Эти операции отнимают довольно много времени ->>
Поэтому, имея небольшое количество времени и CPU в распоряжении, мы сосредоточились на выделении фичей из текста.
Тем не менее были приняты попытки выделения и анализа ссылок в комментариях, смысловое укорачивание и чистка как комментариев, так и постов.

## Feature engineering
У нас была идея! Фичи - количество словом, пунктуации, букв, символов и прочее - это понятно. Но глядя на частотное распределение стало понятно, что мы мало от них добьемся. Тогда мы подумали! Распределить все эти числовые фичи по постам и проранжировать их.

## Learning to rank
Результаты первых прогонов XGBoostRank и EDA  показали, что наиболее важными для ранжирования являются признаки: количество букв в комментариях, количество  знаков препинания, количество эмодзи. Зависимость эта конечно же не линейна, поскольку везде есть своя золотая середина - даже в самом популярном комментарии. 
Cредний ncdg был около 0.8050942926890668, диапазон значений - от 0.6 до 0.99.vТогда мы сократили признаки и оставили только самые оптимальные.

Как итог: валидация не ухудшилась и не улучшилась. Но мы отобрали значимые фичи и с 56 сократили их до 3. Т. е. уменьшили затраты на модель.
