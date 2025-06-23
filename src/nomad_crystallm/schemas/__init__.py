from nomad.config.models.plugins import SchemaPackageEntryPoint


class CrystallmSchemaPackageEntryPoint(SchemaPackageEntryPoint):
    def load(self):
        from nomad_crystallm.schemas.schema import m_package

        return m_package


crystallm_schemas = CrystallmSchemaPackageEntryPoint(
    name='Crystallm Schema',
    description='Schema for running CrystaLLM on NOMAD deployments.',
)
