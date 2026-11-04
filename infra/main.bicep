// Agentic Knowledge Search のインフラ一式。
// デプロイ例:
//   az group create -n rg-aks-demo -l japaneast
//   az deployment group create -g rg-aks-demo -f infra/main.bicep -p infra/main.parameters.json

@description('リソース名の先頭につける文字列（小文字英数字）')
@minLength(3)
@maxLength(12)
param prefix string = 'akdemo'

@description('Container Apps などを置くリージョン')
param location string = resourceGroup().location

@description('Static Web Apps を置くリージョン（Free プランは対応リージョンが限られる）')
param staticWebAppLocation string = 'eastasia'

@description('バックエンドのコンテナイメージ（例: ghcr.io/<user>/agentic-knowledge-search-api:latest）')
param containerImage string

@description('API を呼べるフロントエンドの URL（カンマ区切り）')
param allowedOrigins string = ''

@description('IP ごとの 1 日あたりの上限回数')
param dailyRequestLimit int = 30

@description('チャットモデルの容量（1000 トークン/分単位）')
param chatCapacity int = 20

@description('埋め込みモデルの容量（1000 トークン/分単位）')
param embeddingCapacity int = 20

var suffix = uniqueString(resourceGroup().id)
var searchIndex = 'knowledge'

resource logs 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: '${prefix}-logs-${suffix}'
  location: location
  properties: {
    sku: { name: 'PerGB2018' }
    retentionInDays: 30
  }
}

resource openAi 'Microsoft.CognitiveServices/accounts@2024-10-01' = {
  name: '${prefix}-aoai-${suffix}'
  location: location
  kind: 'OpenAI'
  sku: { name: 'S0' }
  properties: {
    customSubDomainName: '${prefix}-aoai-${suffix}'
    publicNetworkAccess: 'Enabled'
  }
}

resource chatModel 'Microsoft.CognitiveServices/accounts/deployments@2024-10-01' = {
  parent: openAi
  name: 'gpt-4.1-mini'
  sku: { name: 'GlobalStandard', capacity: chatCapacity }
  properties: {
    model: { format: 'OpenAI', name: 'gpt-4.1-mini', version: '2025-04-14' }
  }
}

resource embeddingModel 'Microsoft.CognitiveServices/accounts/deployments@2024-10-01' = {
  parent: openAi
  name: 'text-embedding-3-small'
  sku: { name: 'GlobalStandard', capacity: embeddingCapacity }
  properties: {
    model: { format: 'OpenAI', name: 'text-embedding-3-small', version: '1' }
  }
  dependsOn: [chatModel] // 同じアカウントへの同時デプロイを避ける
}

resource search 'Microsoft.Search/searchServices@2023-11-01' = {
  name: '${prefix}-search-${suffix}'
  location: location
  sku: { name: 'free' }
  properties: {
    replicaCount: 1
    partitionCount: 1
  }
}

resource env 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: '${prefix}-env-${suffix}'
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logs.properties.customerId
        sharedKey: logs.listKeys().primarySharedKey
      }
    }
  }
}

resource api 'Microsoft.App/containerApps@2024-03-01' = {
  name: '${prefix}-api'
  location: location
  properties: {
    managedEnvironmentId: env.id
    configuration: {
      ingress: {
        external: true
        targetPort: 8000
        transport: 'auto'
      }
      secrets: [
        { name: 'openai-key', value: openAi.listKeys().key1 }
        { name: 'search-key', value: search.listAdminKeys().primaryKey }
      ]
    }
    template: {
      containers: [
        {
          name: 'api'
          image: containerImage
          resources: { cpu: json('0.5'), memory: '1Gi' }
          env: [
            { name: 'AZURE_OPENAI_ENDPOINT', value: openAi.properties.endpoint }
            { name: 'AZURE_OPENAI_API_KEY', secretRef: 'openai-key' }
            { name: 'AZURE_OPENAI_CHAT_DEPLOYMENT', value: chatModel.name }
            { name: 'AZURE_OPENAI_EMBEDDING_DEPLOYMENT', value: embeddingModel.name }
            { name: 'AZURE_SEARCH_ENDPOINT', value: 'https://${search.name}.search.windows.net' }
            { name: 'AZURE_SEARCH_API_KEY', secretRef: 'search-key' }
            { name: 'AZURE_SEARCH_INDEX', value: searchIndex }
            { name: 'DAILY_REQUEST_LIMIT', value: string(dailyRequestLimit) }
            { name: 'ALLOWED_ORIGINS', value: allowedOrigins }
          ]
          probes: [
            {
              type: 'Liveness'
              httpGet: { path: '/health', port: 8000 }
            }
          ]
        }
      ]
      scale: {
        minReplicas: 0 // 使われていないときは 0 台にして費用を抑える
        maxReplicas: 1 // 回数制限をメモリで数えているため 1 台に固定
      }
    }
  }
}

resource web 'Microsoft.Web/staticSites@2023-12-01' = {
  name: '${prefix}-web-${suffix}'
  location: staticWebAppLocation
  sku: { name: 'Free', tier: 'Free' }
  properties: {}
}

output apiUrl string = 'https://${api.properties.configuration.ingress.fqdn}'
output webUrl string = 'https://${web.properties.defaultHostname}'
output staticWebAppName string = web.name
output openAiEndpoint string = openAi.properties.endpoint
output searchEndpoint string = 'https://${search.name}.search.windows.net'
