from wiki_database import WikiDatabaseSqlite
from utils_db import mkdir_if_not_exist
import config
import os

if __name__ == '__main__':
	path_wiki_pages = os.path.join(config.ROOT, config.DATA_DIR, config.WIKI_PAGES_DIR, 'wiki-pages')
	path_wiki_database_dir = os.path.join(config.ROOT, config.DATA_DIR, config.DATABASE_DIR)
	# mkdir_if_not_exist(path_wiki_database_dir)
	wiki_database = WikiDatabaseSqlite(path_wiki_database_dir, path_wiki_pages)
