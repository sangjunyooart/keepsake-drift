# rss_sources.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class RssSource:
    key: str           # api_sources.source_key
    name: str          # human readable
    url: str           # RSS feed URL
    trust: float = 0.9
    lang: str = "en"
    tags: Optional[List[str]] = None


# 공용 피드 풀: 안정적인 RSS 위주로 시작
# (원하면 더 추가해도 됨)
RSS_SOURCES: List[RssSource] = [
    # ── World news (human-time, liminal-time) ──
    RssSource(key="nyt_rss_world", name="NYT (World)",
              url="https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
              trust=0.9, tags=["world", "news"]),
    RssSource(key="guardian_rss_world", name="The Guardian (World)",
              url="https://www.theguardian.com/world/rss",
              trust=0.85, tags=["world", "news"]),
    RssSource(key="aljazeera_rss", name="Al Jazeera English",
              url="https://www.aljazeera.com/xml/rss/all.xml",
              trust=0.8, tags=["world", "news"]),
    RssSource(key="bbc_rss_world", name="BBC (World)",
              url="https://feeds.bbci.co.uk/news/world/rss.xml",
              trust=0.85, tags=["world", "news"]),

    # ── Science (all temporalities) ──
    RssSource(key="nyt_rss_science", name="NYT (Science)",
              url="https://rss.nytimes.com/services/xml/rss/nyt/Science.xml",
              trust=0.9, tags=["science"]),
    RssSource(key="bbc_rss_science", name="BBC (Science & Environment)",
              url="https://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
              trust=0.85, tags=["science", "environment"]),
    RssSource(key="sciencedaily_rss", name="ScienceDaily",
              url="https://www.sciencedaily.com/rss/all.xml",
              trust=0.85, tags=["science", "environment"]),
    RssSource(key="nature_rss", name="Nature (journal)",
              url="https://www.nature.com/nature.rss",
              trust=0.9, tags=["science"]),
    RssSource(key="livescience_rss", name="Live Science",
              url="https://www.livescience.com/feeds/all",
              trust=0.8, tags=["science"]),
    RssSource(key="phys_rss", name="Phys.org",
              url="https://phys.org/rss-feed/",
              trust=0.8, tags=["science", "environment"]),

    # ── Environment & climate (environment-time) ──
    RssSource(key="nyt_rss_climate", name="NYT (Climate)",
              url="https://rss.nytimes.com/services/xml/rss/nyt/Climate.xml",
              trust=0.9, tags=["environment", "climate"]),
    RssSource(key="guardian_rss_environment", name="The Guardian (Environment)",
              url="https://www.theguardian.com/environment/rss",
              trust=0.85, tags=["environment", "climate"]),
    RssSource(key="guardian_rss_biodiversity", name="The Guardian (Biodiversity)",
              url="https://www.theguardian.com/environment/biodiversity/rss",
              trust=0.8, tags=["environment", "wildlife"]),
    RssSource(key="mongabay_rss", name="Mongabay",
              url="https://news.mongabay.com/feed/",
              trust=0.8, tags=["environment", "wildlife"]),
    RssSource(key="eos_rss", name="EOS (AGU Earth Science)",
              url="https://eos.org/feed",
              trust=0.85, tags=["environment", "science"]),

    # ── Wildlife & more-than-human (more_than_human-time) ──
    RssSource(key="sciencedaily_bio", name="ScienceDaily (Biodiversity)",
              url="https://www.sciencedaily.com/rss/earth_climate/biodiversity.xml",
              trust=0.8, tags=["wildlife", "environment"]),
    RssSource(key="sciencedaily_animals", name="ScienceDaily (Animals)",
              url="https://www.sciencedaily.com/rss/plants_animals.xml",
              trust=0.8, tags=["wildlife"]),
    RssSource(key="conservation_intl", name="Conservation International",
              url="https://www.conservation.org/blog/rss-feeds",
              trust=0.8, tags=["wildlife", "environment"]),
    RssSource(key="undark_rss", name="Undark Magazine",
              url="https://undark.org/feed/",
              trust=0.8, tags=["science", "wildlife"]),

    # ── Technology & digital (digital-time) ──
    RssSource(key="nyt_rss_technology", name="NYT (Technology)",
              url="https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml",
              trust=0.9, tags=["tech"]),
    RssSource(key="guardian_rss_tech", name="The Guardian (Technology)",
              url="https://www.theguardian.com/technology/rss",
              trust=0.85, tags=["tech"]),
    RssSource(key="ars_rss", name="Ars Technica",
              url="https://feeds.arstechnica.com/arstechnica/index",
              trust=0.8, tags=["tech", "science"]),
    RssSource(key="wired_rss", name="Wired",
              url="https://www.wired.com/feed/rss",
              trust=0.8, tags=["tech"]),

    # ── Culture & human interest (human-time) ──
    RssSource(key="nyt_rss_arts", name="NYT (Arts)",
              url="https://rss.nytimes.com/services/xml/rss/nyt/Arts.xml",
              trust=0.85, tags=["culture", "human"]),
    RssSource(key="guardian_rss_culture", name="The Guardian (Culture)",
              url="https://www.theguardian.com/culture/rss",
              trust=0.8, tags=["culture", "human"]),
    RssSource(key="smithsonian_rss", name="Smithsonian Magazine",
              url="https://www.smithsonianmag.com/rss/latest_articles/",
              trust=0.85, tags=["culture", "science", "human"]),
    RssSource(key="cosmos_rss", name="Cosmos Magazine",
              url="https://cosmosmagazine.com/feed",
              trust=0.8, tags=["science", "culture"]),

    # ── Middle East / regional ──
    RssSource(key="guardian_rss_middleeast", name="The Guardian (Middle East)",
              url="https://www.theguardian.com/world/middleeast/rss",
              trust=0.8, tags=["world", "middleeast"]),

    # ── Infrastructure & economy (infrastructure-time) ──
    RssSource(key="bbc_rss_business", name="BBC (Business)",
              url="https://feeds.bbci.co.uk/news/business/rss.xml",
              trust=0.85, tags=["economy", "infrastructure"]),
    RssSource(key="nyt_rss_economy", name="NYT (Economy)",
              url="https://rss.nytimes.com/services/xml/rss/nyt/Economy.xml",
              trust=0.85, tags=["economy", "infrastructure"]),
]


def sources_by_tag(tag: str) -> List[RssSource]:
    out: List[RssSource] = []
    for s in RSS_SOURCES:
        if s.tags and tag in s.tags:
            out.append(s)
    return out