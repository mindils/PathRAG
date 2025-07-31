import re
import os
from typing import List, Dict


class DocumentPreprocessor:
    """Предобработка корпоративных документов перед индексацией"""

    def __init__(self):
        self.section_patterns = {
            'приказ': r'ПРИКАЗ\s*\n.*?(?=\n\n)',
            'раздел': r'^\d+\.\s+[А-ЯЁ][^\n]+',
            'пункт': r'^\d+\.\d+\.?\s+',
            'подпункт': r'^\d+\.\d+\.\d+\.?\s+',
            'определение': r'[-–—]\s*([^;]+?)\s*[-–—]\s*([^;]+)',
        }

    def preprocess_document(self, text: str) -> Dict[str, any]:
        """Предобработка одного документа"""

        # Извлечение метаданных
        metadata = self.extract_metadata(text)

        # Разделение на логические секции
        sections = self.split_into_sections(text)

        # Обогащение секций контекстом
        enriched_sections = self.enrich_sections(sections, metadata)

        return {
            'metadata': metadata,
            'sections': enriched_sections,
            'full_text': text
        }

    def extract_metadata(self, text: str) -> Dict[str, str]:
        """Извлечение метаданных документа"""
        metadata = {}

        # Извлечение номера и даты приказа
        order_match = re.search(r'ПРИКАЗ\s*\n\s*.*?(\d{1,2})\s+(\w+)\s+(\d{4})\s*г\.\s*№\s*(\d+)', text)
        if order_match:
            metadata['order_date'] = f"{order_match.group(1)} {order_match.group(2)} {order_match.group(3)}"
            metadata['order_number'] = order_match.group(4)

        # Извлечение названия документа
        title_match = re.search(r'(?:Об утверждении|О введении в действие)\s+(.+?)(?:\n|$)', text)
        if title_match:
            metadata['title'] = title_match.group(1).strip()

        # Дата вступления в силу
        effective_match = re.search(r'(?:ввести в действие|вступает в силу)\s+с\s+(\d{1,2}\.\d{1,2}\.\d{4})', text)
        if effective_match:
            metadata['effective_date'] = effective_match.group(1)

        return metadata

    def split_into_sections(self, text: str) -> List[Dict[str, str]]:
        """Разделение документа на логические секции"""
        sections = []

        # Разделение по основным разделам
        lines = text.split('\n')
        current_section = None
        current_content = []

        for line in lines:
            # Проверка на начало нового раздела
            if re.match(r'^\d+\.\s+[А-ЯЁ]', line):
                if current_section:
                    sections.append({
                        'title': current_section,
                        'content': '\n'.join(current_content),
                        'level': 'section'
                    })
                current_section = line.strip()
                current_content = []
            else:
                current_content.append(line)

        # Добавление последней секции
        if current_section:
            sections.append({
                'title': current_section,
                'content': '\n'.join(current_content),
                'level': 'section'
            })

        return sections

    def enrich_sections(self, sections: List[Dict], metadata: Dict) -> List[Dict]:
        """Обогащение секций контекстной информацией"""
        enriched = []

        for section in sections:
            enriched_section = section.copy()

            # Добавление контекста документа
            context = f"Документ: {metadata.get('title', 'Неизвестный документ')}"
            if 'order_number' in metadata:
                context += f" (Приказ № {metadata['order_number']} от {metadata['order_date']})"
            if 'effective_date' in metadata:
                context += f". Действует с {metadata['effective_date']}"

            enriched_section['context'] = context
            enriched.append(enriched_section)

        return enriched