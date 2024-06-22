from typing import Optional

class Path:
    def plots(self, path: Optional[str]):
        return self.storage(f'plots/{path}')

    def board(self, path: Optional[str]):
        return self.storage(f'board/{path}')

    def logs(self, path: Optional[str]):
        return self.storage(f'logs/{path}')

    def storage(self, path: Optional[str]):
        path = self.clean_path(path)

        return f'storage{path}'

    def resources(self, path: Optional[str]):
        path = self.clean_path(path)

        return f'resources{path}'

    def clean_path(self, path: Optional[str]):
        if path is None:
            return ''

        if path.endswith('/') is True:
            path = path[:-1]

        if path.startswith('/') is True:
            return path

        return f'/{path}'

path = Path()
