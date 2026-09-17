from pydantic import BaseModel, ConfigDict


class SubjectOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    code: str
    name: str
    branch: str
    scheme: str
    semester: int


class ModuleOut(BaseModel):
    number: int
    title: str | None
    notes: int = 0
    indexed_notes: int = 0


class SubjectModulesOut(BaseModel):
    subject: SubjectOut
    modules: list[ModuleOut]
