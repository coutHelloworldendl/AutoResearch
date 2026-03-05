import requests
from bs4 import BeautifulSoup
import json
import os
import time
from typing import Dict, Optional, List
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import datetime
import openreview
import getpass
from tqdm import tqdm

INFO = json.load(open("info.json", "r")) 
USERNAME = os.getenv("OR_USERNAME", None)
PASSWORD = os.getenv("OR_PASSWORD", None)

class Requester:
    def __init__(self, timeout: int, suc_interval: float, fail_interval: float, max_retries: int):
        self.headers = { "User-Agent": "curl/8.4.0" }
        self.timeout = timeout
        self.suc_interval = suc_interval
        self.fail_interval = fail_interval
        self.max_retries = max_retries

    def request_once(self, url: str) -> dict:
        try:
            response = requests.get(url, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            print(f"[ERROR] Request to {url} failed: {e}")
            raise
    
    def request(self, url: str) -> dict:
        for _ in range(self.max_retries):
            try:
                response = self.request_once(url)
                time.sleep(self.suc_interval)
                return response
            except requests.RequestException as e:
                print(f"[WARNING] Request to {url} failed: {e}. Retrying in {self.fail_interval} seconds...")
                time.sleep(self.fail_interval)
                
        print(f"[ERROR] All retries failed for {url}.")
        raise requests.RequestException(f"Failed to fetch {url} after {self.max_retries} retries.")

class Paper:
    """A class to hold paper information."""
    def __init__(self):
        self.title = None
        self.authors = None
        self.paper_id = None
        self.pdf_url = None
        self.abstract = None
    
    def to_dict(self) -> dict:
        return {
            'paper_id': self.paper_id,
            'title': self.title,
            'authors': self.authors,
            'pdf_url': self.pdf_url,
            'abstract': self.abstract,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Paper':
        paper = cls()
        paper.title = data.get('title', None) if paper.title is None else paper.title
        paper.authors = data.get('authors', None) if paper.authors is None else paper.authors
        paper.paper_id = data.get('paper_id', None) if paper.paper_id is None else paper.paper_id
        paper.pdf_url = data.get('pdf_url', None) if paper.pdf_url is None else paper.pdf_url
        paper.abstract = data.get('abstract', None) if paper.abstract is None else paper.abstract
        return paper
    
class Crawler:
    def __init__(self, conference_name, year, timeout: int = 10, suc_interval: float = 1.0, fail_interval: float = 2.0, max_retries: int = 3):
        assert(conference_name in INFO), f"Conference {conference_name} not found in INFO."
        self.conference_name = conference_name
        self.year = year
        self.url = INFO[conference_name][str(year)]
        self.requester = Requester(timeout=timeout, suc_interval=suc_interval, fail_interval=fail_interval, max_retries=max_retries)
        self.output = 'paper/' + conference_name + str(year) + '.json'

    def save_papers(self, papers: List[Paper]):
        data = [paper.to_dict() for paper in papers]
        with open(self.output, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"[INFO] Saved {len(papers)} papers to {self.output}")

    def parser(self, url: str) -> List[Paper]:
        if self.conference_name == "ACL":
            return self.ACL_parser(url)
        elif self.conference_name == "EMNLP":
            return self.EMNLP_parser(url)
        elif self.conference_name == "ICLR":
            return self.ICLR_parser(url)
        elif self.conference_name == "CVPR":
            return self.CVPR_parser(url)
        elif self.conference_name == "ICML":
            return self.ICML_parser(url)
        elif self.conference_name == "ICCV":
            return self.ICCV_parser(url)
        elif self.conference_name == "AAAI":
            return self.AAAI_parser(url)
        elif self.conference_name == "NIPS":
            return self.NIPS_parser(url)
        else:
            raise ValueError(f"Parser for conference {self.conference_name} is not implemented.")

    def ACL_parser(self, url: str) -> List[Paper]:
        # Get HTML response
        assert self.conference_name == "ACL", "ACL_parser is only for ACL conference."
        soup = BeautifulSoup(self.requester.request(url=url), 'html.parser')

        # Extract paper elements
        raw_papers = [p for p in soup.find('div', id=f'{self.year}acl-long').find_all('p', class_='d-sm-flex align-items-stretch') if p.select_one('a.badge-primary')]
        assert len(raw_papers) > 0, f"No papers found for ACL {self.year}."

        papers = []
        for i, p_element in enumerate(raw_papers, 1):
            # 1. Extract Title
            try:
                title = p_element.select_one('span.d-block > strong > a').text.strip()
            except Exception as e:
                print(f"[WARNING] Could not extract title for paper {i}: {e}")
                continue

            # 2. Extract Authors
            try:
                authors = [author.text.strip() for author in p_element.select('a[href^="/people/"]')]
            except Exception as e:
                print(f"[WARNING] Could not extract authors for paper {i}: {e}")
                authors = []

            # 3. Extract PDF URL
            try:
                pdf_tag = p_element.select_one('a.badge-primary[href$=".pdf"]')['href']
            except Exception as e:
                print(f"[WARNING] Could not extract PDF URL for paper {i}: {e}")
                pdf_tag = None
            
            # 4. Extract Abstract
            try:
                abstract = p_element.find_next_sibling('div', class_='abstract-collapse').find('div', class_='card-body').text.strip()
            except Exception as e:
                print(f"[WARNING] Could not extract abstract for paper {i}: {e}")
                abstract = None

            paper = Paper().from_dict({
                'paper_id': "ACL" + str(self.year) + f"-{i:05d}",
                'title': title,
                'authors': authors,
                'pdf_url': pdf_tag,
                'abstract': abstract
            })

            papers.append(paper)
        
        return papers
    
    def EMNLP_parser(self, url: str) -> List[Paper]:
        assert self.conference_name == "EMNLP", "EMNLP_parser is only for EMNLP conference."
        # Get HTML response
        soup = BeautifulSoup(self.requester.request(url=url), 'html.parser')

        # Extract paper elements
        raw_papers = [p for p in soup.find('div', id=f'{self.year}emnlp-main').find_all('p', class_='d-sm-flex align-items-stretch') if p.select_one('a.badge-primary')]
        assert len(raw_papers) > 0, f"No papers found for EMNLP {self.year}."

        papers = []
        for i, p_element in enumerate(raw_papers, 1):
            # 1. Extract Title
            try:
                title = p_element.select_one('span.d-block > strong > a').text.strip()
            except Exception as e:
                print(f"[WARNING] Could not extract title for paper {i}: {e}")
                continue

            # 2. Extract Authors
            try:
                authors = [author.text.strip() for author in p_element.select('a[href^="/people/"]')]
            except Exception as e:
                print(f"[WARNING] Could not extract authors for paper {i}: {e}")
                authors = []

            # 3. Extract PDF URL
            try:
                pdf_tag = p_element.select_one('a.badge-primary[href$=".pdf"]')['href']
            except Exception as e:
                print(f"[WARNING] Could not extract PDF URL for paper {i}: {e}")
                pdf_tag = None
            
            # 4. Extract Abstract
            try:
                abstract = p_element.find_next_sibling('div', class_='abstract-collapse').find('div', class_='card-body').text.strip()
            except Exception as e:
                print(f"[WARNING] Could not extract abstract for paper {i}: {e}")
                abstract = None

            paper = Paper().from_dict({
                'paper_id': "EMNLP" + str(self.year) + f"-{i:05d}",
                'title': title,
                'authors': authors,
                'pdf_url': pdf_tag,
                'abstract': abstract
            })

            papers.append(paper)
        
        return papers

    def ICLR_parser(self, url: str) -> List[Paper]:
        assert self.conference_name == "ICLR", "ICLR_parser is only for ICLR conference."
        client = openreview.api.OpenReviewClient(
            baseurl='https://api2.openreview.net',
            username=input("Enter your OpenReview username: ") if not USERNAME else USERNAME,
            password=getpass.getpass("Enter your OpenReview password: ") if not PASSWORD else PASSWORD,
            )
        raw_papers = client.get_all_notes(invitation=url)
        assert len(raw_papers) > 0, f"No papers found for ICLR {self.year}."

        papers = []
        for idx, paper in enumerate(raw_papers, 1):

            # 1. Extract Title
            title = paper.content.get('title', {}).get('value', '') if isinstance(paper.content.get('title'), dict) else paper.content.get('title', '')

            # 2. Extract Authors
            authors_content = paper.content.get('authors', {})
            if isinstance(authors_content, dict):
                authors = ', '.join(authors_content.get('value', []))
            else:
                authors = ', '.join(authors_content if isinstance(authors_content, list) else [])

            # 3. Extract PDF URL
            pdf_url = f'https://openreview.net/pdf?id={paper.id}'

            # 4. Extract Abstract
            abstract = paper.content.get('abstract', {}).get('value', '') if isinstance(paper.content.get('abstract'), dict) else paper.content.get('abstract', '')

            paper = Paper().from_dict({
                'paper_id': "ICLR" + str(self.year) + f"-{idx:05d}",
                'title': title,
                'authors': authors,
                'pdf_url': pdf_url,
                'abstract': abstract
            })

            if hasattr(paper, 'details') and paper.details and 'directReplies' in paper.details:
                if "Decision" in str(paper.details) and "Accept" in str(paper.details):
                    papers.append(paper)

        return papers
               
    def CVPR_parser(self, url: str) -> List[Paper]:
        assert self.conference_name == "CVPR", "CVPR_parser is only for CVPR conference."
        soup = BeautifulSoup(self.requester.request(url=url), 'html.parser')

        papers = []
        raw_papers = soup.find(id="content").find("dl").find_all(recursive=False)[1:-1]
        raw_papers = [raw_papers[i:i+3] for i in range(0, len(raw_papers), 3)]
        for idx, raw_paper in tqdm(enumerate(raw_papers, 1)):
            
            # 1. Extract Title
            title = raw_paper[0].find("a").text.strip()

            # 2. Extract Authors
            authors = [author.strip() for author in raw_paper[1].text.strip().split(",")]

            # 3. Extract PDF URL
            pdf_url = raw_paper[2].find("a", string="pdf")["href"]

            # 4. Extract Abstract
            abs_output = self.requester.request(url="https://openaccess.thecvf.com/"+raw_paper[0].find("a")["href"])
            try:
                abs_soup = BeautifulSoup(abs_output, 'html.parser')
            except Exception as e:
                print(f"[WARNING] Could not parse abstract HTML for paper {idx}: {e}")
                abstract = ""
            else:
                abstract = abs_soup.find("div", id="abstract").text.strip()

            paper = Paper().from_dict({
                'paper_id': "CVPR" + str(self.year) + f"-{idx:05d}",
                'title': title,
                'authors': authors,
                'pdf_url': pdf_url,
                'abstract': abstract
            })

            papers.append(paper)

        return papers

    def ICML_parser(self, url: str) -> List[Paper]:
        assert self.conference_name == "ICML", "ICML_parser is only for ICML conference."
        soup = BeautifulSoup(self.requester.request(url=url), 'html.parser')

        papers = []
        for idx, paper_div in tqdm(enumerate(soup.select("div.paper"), 1)):
            title = paper_div.select_one("p.title").get_text(strip=True)
            authors = paper_div.select_one("span.authors").get_text(strip=True)
            pdf_link = paper_div.select_one('a[href$=".pdf"]')["href"]
            abs_link = paper_div.select_one('a[href*=".html"]')["href"]

            response = self.requester.request(url=abs_link)
            abs_soup = BeautifulSoup(response, 'html.parser')
            abstract_tag = abs_soup.select_one("div.abstract") or abs_soup.select_one("p.abstract")
            if abstract_tag:
                abstract = abstract_tag.get_text(strip=True)
            else:
                print(f"Abstract not found for paper: {title}")
                abstract = None

            paper = Paper().from_dict({
                'paper_id': "ICML" + str(self.year) + f"-{idx:05d}",
                'title': title,
                'authors': authors,
                'pdf_url': pdf_link,
                'abstract': abstract
            })

            papers.append(paper)

        return papers

    def ICCV_parser(self, url: str) -> List[Paper]:
        assert self.conference_name == "ICCV", "ICCV_parser is only for ICCV conference."
        soup = BeautifulSoup(self.requester.request(url=url), 'html.parser')

        papers = []
        raw_papers = soup.find(id="content").find("dl").find_all(recursive=False)[1:-1]
        raw_papers = [raw_papers[i:i+3] for i in range(0, len(raw_papers), 3)]
        for idx, raw_paper in tqdm(enumerate(raw_papers, 1)):
            
            # 1. Extract Title
            title = raw_paper[0].find("a").text.strip()

            # 2. Extract Authors
            authors = [author.strip() for author in raw_paper[1].text.strip().split(",")]

            # 3. Extract PDF URL
            pdf_url = raw_paper[2].find("a", string="pdf")["href"]

            # 4. Extract Abstract
            abs_output = self.requester.request(url="https://openaccess.thecvf.com/"+raw_paper[0].find("a")["href"])
            try:
                abs_soup = BeautifulSoup(abs_output, 'html.parser')
            except Exception as e:
                print(f"[WARNING] Could not parse abstract HTML for paper {idx}: {e}")
                abstract = ""
            else:
                abstract = abs_soup.find("div", id="abstract").text.strip()

            paper = Paper().from_dict({
                'paper_id': "ICCV" + str(self.year) + f"-{idx:05d}",
                'title': title,
                'authors': authors,
                'pdf_url': pdf_url,
                'abstract': abstract
            })

            papers.append(paper)

        return papers

    def AAAI_parser(self, url: str) -> List[Paper]:
        assert self.conference_name == "AAAI", "AAAI_parser is only for AAAI conference."
        soup = BeautifulSoup(self.requester.request(url=url), 'html.parser')

        papers = []
        part_urls = soup.find("div", class_="archive-description taxonomy-archive-description taxonomy-description").find_all("a", href=True)
        for part_url in part_urls:
            part_soup = BeautifulSoup(self.requester.request(url=part_url['href']), 'html.parser')
            raw_papers = part_soup.find_all("div", class_="obj_article_summary")
            for idx, raw_paper in tqdm(enumerate(raw_papers, 1)):
                
                # 1. Extract Title
                title = raw_paper.find("h3", class_="title").find("a").text.strip()

                # 2. Extract Authors
                authors = [raw_paper.find("div", class_="authors").text.strip().split(",")]

                # 3. Extract PDF URL
                pdf_url = raw_paper.find("a", class_="obj_galley_link pdf")["href"]

                # 4. Extract Abstract
                abs_output = self.requester.request(url=raw_paper.find("h3", class_="title").find("a")["href"])
                abs_soup = BeautifulSoup(abs_output, 'html.parser')
                abstract = abs_soup.find("section", class_="item abstract").text.strip().replace("Abstract\n", "").lstrip()

                paper = Paper().from_dict({
                    'paper_id': "AAAI" + str(self.year) + f"-{idx:05d}",
                    'title': title,
                    'authors': authors,
                    'pdf_url': pdf_url,
                    'abstract': abstract
                })

                papers.append(paper)

        return papers

    def NIPS_parser(self, url: str) -> List[Paper]:
        assert self.conference_name == "NIPS", "NIPS_parser is only for NIPS conference."
        try:
            client = openreview.api.OpenReviewClient(
                baseurl='https://api2.openreview.net',
                username=input("Enter your OpenReview username: ") if not USERNAME else USERNAME,
                password=getpass.getpass("Enter your OpenReview password: ") if not PASSWORD else PASSWORD,
            )
            raw_papers = client.get_all_notes(invitation=url)

            papers = []
            for idx, paper in enumerate(raw_papers, 1):

                if "Submitted" in paper.content.get("venue").get("value", "N/A"):
                    continue

                # 1. Extract Title
                title = paper.content.get('title', {}).get('value', '') if isinstance(paper.content.get('title'), dict) else paper.content.get('title', '')

                # 2. Extract Authors
                authors_content = paper.content.get('authors', {})
                if isinstance(authors_content, dict):
                    authors = ', '.join(authors_content.get('value', []))
                else:
                    authors = ', '.join(authors_content if isinstance(authors_content, list) else [])

                # 3. Extract PDF URL
                pdf_url = f'https://openreview.net/pdf?id={paper.id}'

                # 4. Extract Abstract
                abstract = paper.content.get('abstract', {}).get('value', '') if isinstance(paper.content.get('abstract'), dict) else paper.content.get('abstract', '')

                paper = Paper().from_dict({
                    'paper_id': "NIPS" + str(self.year) + f"-{idx:05d}",
                    'title': title,
                    'authors': authors,
                    'pdf_url': pdf_url,
                    'abstract': abstract
                })
        
                papers.append(paper)

            return papers
        
        except Exception as e:
            print(f"[ERROR] Failed to parse NIPS papers: {e}")
            return []