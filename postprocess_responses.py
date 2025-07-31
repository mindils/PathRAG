class ResponsePostprocessor:
    """Постобработка ответов для корпоративных документов"""

    def __init__(self):
        self.important_terms = {
            'обязан', 'должен', 'необходимо', 'требуется',
            'не позднее', 'в течение', 'до', 'после',
            'запрещается', 'не допускается'
        }

    def postprocess_response(self, response: str, query: str) -> str:
        """Постобработка ответа"""

        # Выделение важных моментов
        response = self.highlight_important_info(response)

        # Добавление дисклеймера при необходимости
        response = self.add_disclaimer_if_needed(response, query)

        # Форматирование для читаемости
        response = self.format_response(response)

        return response

    def highlight_important_info(self, response: str) -> str:
        """Выделение важной информации"""

        # Выделение сроков
        response = re.sub(
            r'(\d+\s*(?:дней|месяцев|лет|календарных дней))',
            r'**\1**',
            response
        )

        # Выделение сумм
        response = re.sub(
            r'(\d+(?:\s*\d{3})*(?:\.\d{2})?\s*рублей?)',
            r'**\1**',
            response
        )

        # Выделение важных терминов
        for term in self.important_terms:
            response = re.sub(
                f'({term})',
                r'**\1**',
                response,
                flags=re.IGNORECASE
            )

        return response

    def add_disclaimer_if_needed(self, response: str, query: str) -> str:
        """Добавление дисклеймера при необходимости"""

        keywords = ['последние изменения', 'актуальн', 'действующ']

        if any(keyword in query.lower() for keyword in keywords):
            disclaimer = "\n\n⚠️ **Внимание**: Приведенная информация основана на документах, имеющихся в базе. Для получения самой актуальной информации рекомендуется обратиться в Департамент управления персоналом."
            response += disclaimer

        return response

    def format_response(self, response: str) -> str:
        """Форматирование ответа для улучшения читаемости"""

        # Разделение на абзацы по логическим блокам
        response = re.sub(r'\n{3,}', '\n\n', response)

        # Добавление эмодзи для разделов
        section_mapping = {
            'Основные положения': '📋',
            'Необходимые документы': '📄',
            'Сроки': '⏰',
            'Суммы и размеры': '💰',
            'Важно': '⚠️',
            'Порядок действий': '📌'
        }

        for section, emoji in section_mapping.items():
            response = response.replace(section, f'{emoji} {section}')

        return response