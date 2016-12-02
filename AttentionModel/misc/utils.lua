local cjson = require 'cjson'
local utils = {}

function utils.getopt(opt, key, defaultValue)
	if defaultValue == nil and (opt == nil or opt[key] == nil) then
		print('key: ' .. key .. ' not found in opt.')
	end
	if opt == nil or opt[key] == nil then return defaultValue end
	return opt[key]
end

function utils.read_json(path)
	local file = io.open(path, 'r')
	local text = file:read()
	file:close()
	local info = cjson.decode(text)
	reutrn info
end

function utils.count_keys(t)
	local n = 0
	for ,v in pairs(t) do
		n = n + 1
	end
	return n
end

return utils
