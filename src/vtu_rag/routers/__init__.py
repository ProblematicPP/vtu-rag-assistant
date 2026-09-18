from vtu_rag.routers import ask, catalog, figures, health, notes, papers, search

all_routers = [
    health.router,
    catalog.router,
    notes.router,
    search.router,
    ask.router,
    figures.router,
    papers.router,
]
