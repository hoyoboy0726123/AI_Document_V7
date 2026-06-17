module.exports = {
  root: true,
  env: { browser: true, es2020: true, node: true },
  extends: [
    'eslint:recommended',
    'plugin:react/recommended',
    'plugin:react/jsx-runtime',
    'plugin:react-hooks/recommended',
  ],
  ignorePatterns: ['dist', 'build', '.eslintrc.cjs'],
  parserOptions: { ecmaVersion: 'latest', sourceType: 'module' },
  settings: { react: { version: 'detect' } },
  plugins: ['react-refresh'],
  rules: {
    // This is a plain-JS app that does not use PropTypes for prop validation.
    'react/prop-types': 'off',
    // Allow `const { x: _, ...rest } = obj` to intentionally omit a key.
    'no-unused-vars': ['error', { ignoreRestSiblings: true }],
    // Allow `while (true) { ... break }` for SSE/stream read loops.
    'no-constant-condition': ['error', { checkLoops: false }],
    'react-refresh/only-export-components': [
      'warn',
      { allowConstantExport: true },
    ],
  },
}
