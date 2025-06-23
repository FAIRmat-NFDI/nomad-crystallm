from nomad.config.models.plugins import APIEntryPoint


class CrystaLLMAPIEntryPoint(APIEntryPoint):
    def load(self):
        from nomad_crystallm.apis.crystallm import app

        return app


crystallm_api = CrystaLLMAPIEntryPoint(
    prefix='crystallm',
    name='CrystaLLM API',
    description='CrystaLLM custom API.',
)
