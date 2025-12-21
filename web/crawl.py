import web
import argparse

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Crawl conference papers")
    parser.add_argument("--conference", type=str, required=True, help="Conference name")
    parser.add_argument("--year", type=int, required=True, help="Year of the conference")
    parser.add_argument("--timeout", type=int, default=10, help="Request timeout in seconds")
    parser.add_argument("--suc_interval", type=float, default=0.2, help="Interval between successful requests in seconds")
    parser.add_argument("--fail_interval", type=float, default=2.0, help="Interval between failed requests in seconds")
    parser.add_argument("--max_retries", type=int, default=3, help="Maximum number of retries for failed requests")
    args = parser.parse_args()

    conference = args.conference
    year = args.year

    crawler = web.Crawler(conference_name=conference, year=year, timeout=args.timeout, suc_interval=args.suc_interval, fail_interval=args.fail_interval, max_retries=args.max_retries)
    papers = crawler.parser(url=crawler.url)
    crawler.save_papers(papers)