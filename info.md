/mybot
├── bot
│   ├── __init__.py
│   ├── main.py  # основной файл, запускающий бота
│   ├── config.py  # файл с конфигурационными переменными (токены, ключи и т.д.)
│   ├── handlers  # папка с обработчиками сообщений
│   │   ├── __init__.py
│   │   ├── start.py
│   │   ├── text.py
│   │   └── inline_button.py
│   ├── middlewares  # папка с промежуточным программным обеспечением
│   │   ├── __init__.py
│   │   └── example_middleware.py
│   ├── utils  # папка с вспомогательными функциями
│   │   ├── __init__.py
│   │   └── example_util.py
│   └── states  # папка с состояниями для FSM
│       ├── __init__.py
│       └── example_state.py
├── data
│   └── embeddings  # папка с данными для эмбеддингов
│       └── embeddings.pkl
├── Dockerfile
└── README.md
